import numpy as np
from flax import linen as nn
from typing import Sequence
from flax.training import train_state
import jax.numpy as jnp
import jax
import gin
import optax
import tqdm
from pathlib import Path
from models.transformer import TransformerDecoder
from lovecraft.dataset import Dataset, decode_tokens
from models.sample import simple_sample, truncate_sample


def compute_metrics(logits, labels):
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    return {"loss": loss}


@gin.configurable
def make_optimizer(
    learning_rate: float, weight_decay: float
) -> optax.GradientTransformation:
    return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)


@gin.configurable(allowlist=["model"])
def create_train_state(
    init_rng,
    tx: optax.GradientTransformation,
    model: nn.Module,
    input_shape: Sequence[int],
):
    params = model.init(init_rng, jnp.ones(input_shape, dtype=jnp.int32))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state: train_state.TrainState, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, jnp.array(batch["tokens"]))
        loss = optax.softmax_cross_entropy(logits, jnp.array(batch["one_hot"])).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    metrics = compute_metrics(logits, batch["one_hot"])
    state = state.apply_gradients(grads=grads)

    return state, metrics


@jax.jit
def eval_step(state: train_state.TrainState, batch):
    logits = state.apply_fn(state.params, jnp.array(batch["tokens"]))

    metrics = compute_metrics(logits, jnp.array(batch["one_hot"]))
    return metrics


@gin.configurable
def train(num_steps: int, vocab_size: int, random_seed: int, val_freq: int):
    train_novels, val_novels = [], []
    for f in Path("data/lovecraft/dataset").glob("train_encoded_*.npy"):
        train_novels.append(np.load(f))
    for f in Path("data/lovecraft/dataset").glob("val_encoded_*.npy"):
        val_novels.append(np.load(f))

    dataset = Dataset(train_novels, val_novels)

    rng = jax.random.PRNGKey(random_seed)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(
        init_rng,
        tx=make_optimizer(),
        input_shape=[dataset.batch_size, dataset.token_count],
    )
    train_it = dataset.train_iterator()
    train_metrics = {
        "step": [],
        "loss": [],
    }
    val_metrics = {"step": [], "loss": []}

    @jax.jit
    def sample_fn(state, rng):
        return simple_sample(
            state,
            1,
            dataset.token_count,
            dataset.pad_id,
            rng,
        )

    with tqdm.trange(num_steps + 1) as pbar:
        for step in pbar:
            # TODO: use shuffle from jax
            train_batch = next(train_it)
            state, metrics = train_step(state, train_batch)

            train_metrics["step"].append(step)
            train_metrics["loss"].append(metrics["loss"])

            if step % val_freq == 0:
                acc = {"loss": []}

                for val_batch in dataset.val_iterator():
                    metrics = eval_step(state, val_batch)
                    acc["loss"].append(metrics["loss"])

                val_metrics["step"].append(step)
                val_metrics["loss"].append(np.mean(acc["loss"]))

                sample = sample_fn(state, rng)
                truncated_sample = truncate_sample(np.array(sample), dataset.pad_id)
                sample_str = decode_tokens(truncated_sample[0], dataset.vocab_size)
                pbar.write(f"Sample:\n\n{sample_str}\n")

            pbar.set_description(
                f"Step {step}: loss = {train_metrics['loss'][-1].block_until_ready():.4f}, "
                f"val_loss = {val_metrics['loss'][-1]:.4f}"
            )


if __name__ == "__main__":
    gin.external_configurable(
        TransformerDecoder, module="models.transformer", name="TransformerDecoder"
    )
    gin.parse_config_file("data/lovecraft/train_config.gin")
    train()
