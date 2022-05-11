from jax import numpy as jnp, lax
import jax
from flax.training import train_state
import numpy as np


def simple_sample(
    state: train_state.TrainState,
    batch_size: int,
    token_count: int,
    pad_id: int,
    rng: jax.random.PRNGKey,
):
    keys = jax.random.split(rng, batch_size * token_count)
    keys = keys.reshape((batch_size, token_count, 2))

    def body_fn(i, input):
        logits = state.apply_fn(state.params, input, False)
        probs = jax.nn.softmax(logits[:, i])

        def sample_token(probs, key):
            return jax.random.choice(key, probs.shape[-1], p=probs)

        position_keys = keys[:, i]
        sample_token_batch = jax.vmap(sample_token)
        samples = sample_token_batch(probs, position_keys)

        return input.at[:, i].set(samples)

    return lax.fori_loop(
        0,
        token_count,
        body_fn,
        init_val=jnp.ones((batch_size, token_count), dtype=jnp.int32) * pad_id,
    )


def truncate_sample(sample_batch, pad_id):
    truncated = []
    for sample in sample_batch:
        pads = np.where(sample == pad_id)[0]
        if len(pads):
            truncated.append(sample[: pads[0]])
        else:
            truncated.append(sample)

    return truncated
