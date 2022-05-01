from flax import linen as nn
from typing import Callable, Optional
import jax.numpy as jnp

class Attention(nn.Module):
    dimension: int
    mask_queries_from: Optional[int] = None

    WK_init: Callable = nn.initializers.lecun_normal()
    WQ_init: Callable = nn.initializers.lecun_normal()
    WV_init: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, key_x, query_x, value_x):
        assert key_x.shape == query_x.shape
        assert query_x.shape == value_x.shape

        WX_shape = (key_x.shape[-1], self.dimension)
        WK = self.param("WK", self.WK_init, WX_shape)
        WQ = self.param("WQ", self.WQ_init, WX_shape)
        WV = self.param("WV", self.WV_init, WX_shape)

        if self.mask_queries_from is None:
            mask_from_concrete = key_x.shape[-1]
        else:
            assert self.mask_queries_from <= key_x.shape[-1]
            mask_from_concrete = self.mask_queries_from

        mask_zeros = jnp.zeros((1, mask_from_concrete))
        mask_neginf = jnp.full((1, key_x.shape[-1] - mask_from_concrete), -jnp.inf)
        mask = jnp.concatenate([mask_zeros, mask_neginf], axis=1)

        K = jnp.dot(key_x, WK)
        Q = jnp.dot(query_x, WQ)
        V = jnp.dot(value_x, WV)

        logits = jnp.matmul(Q, K.T) / jnp.sqrt(self.dimension)
        weights = nn.softmax(logits + mask)
        return jnp.matmul(weights, V)

        
