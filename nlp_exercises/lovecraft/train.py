import numpy as np
from flax import linen as nn
from typing import Sequence
from flax.training import train_state
import flax
import jax.numpy as jnp
import jax
import gin
import optax
from flax.optim import dynamic_scale
import tqdm
from pathlib import Path
from models.transformer import TransformerDecoder
from lovecraft.dataset import Dataset, decode_tokens
from models.sample import simple_sample, truncate_sample


class TrainStateWithScale(train_state.TrainState):
    ds: dynamic_scale.DynamicScale = None


def compute_metrics(logits, labels):
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    return {"loss": loss}


@gin.configurable
def make_optimizer(
    learning_rate: float, weight_decay: float, warmup_steps: int, total_steps: int
) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-10,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
    )
    return optax.adamw(learning_rate=schedule, weight_decay=weight_decay)


@gin.configurable(allowlist=["model"])
def create_train_state(
    init_rng,
    tx: optax.GradientTransformation,
    model: nn.Module,
    input_shape: Sequence[int],
):
    params = model.init(
        init_rng,
        jnp.ones(input_shape, dtype=jnp.int32),
        False,
    )
    return TrainStateWithScale.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        ds=dynamic_scale.DynamicScale(19),
    )


@jax.jit
def train_step(state: TrainStateWithScale, batch, rng):
    def loss_fn(params):
        logits = state.apply_fn(
            params, jnp.array(batch["tokens"]), True, rngs={"dropout": rng}
        )
        loss = optax.softmax_cross_entropy(logits, jnp.array(batch["one_hot"])).mean()
        return loss, logits

    grad_fn = state.ds.value_and_grad(loss_fn, has_aux=True)
    ds, is_fin, (_, logits), grads = grad_fn(state.params)
    metrics = compute_metrics(logits, batch["one_hot"])
    state = state.apply_gradients(grads=grads)
    state = state.replace(ds=ds)

    return state, metrics


@jax.jit
def eval_step(state: train_state.TrainState, batch):
    logits = state.apply_fn(state.params, jnp.array(batch["tokens"]), False)

    metrics = compute_metrics(logits, jnp.array(batch["one_hot"]))
    return metrics


@gin.configurable
def train(num_steps: int, random_seed: int, val_freq: int):
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
            rng, train_step_rng = jax.random.split(rng)
            state, metrics = train_step(state, train_batch, train_step_rng)

            train_metrics["step"].append(step)
            train_metrics["loss"].append(metrics["loss"])

            if step % val_freq == 0:
                acc = {"loss": []}

                for val_batch in dataset.val_iterator():
                    metrics = eval_step(state, val_batch)
                    acc["loss"].append(metrics["loss"])

                val_metrics["step"].append(step)
                val_metrics["loss"].append(np.mean(acc["loss"]))

                rng, sample_rng = jax.random.split(rng)
                sample = sample_fn(state, sample_rng)
                truncated_sample = truncate_sample(np.array(sample), dataset.pad_id)
                sample_str = decode_tokens(truncated_sample[0], dataset.vocab_size)
                pbar.write(f"\n{sample_str}\n")

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
