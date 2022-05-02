from flax import linen as nn
from typing import Callable, Optional
import jax.numpy as jnp


class MultiHeadAttention(nn.Module):
    dimension: int
    num_heads: int = 1
    mask_queries_from: Optional[int] = None

    WK_init: Callable = nn.initializers.lecun_normal()
    WQ_init: Callable = nn.initializers.lecun_normal()
    WV_init: Callable = nn.initializers.lecun_normal()

    def _split_heads(self, x):
        split = x.reshape((-1, self.num_heads, self.dimension // self.num_heads))
        reorder = split.transpose((1, 0, 2))
        return reorder

    @nn.compact
    def __call__(self, key_x, query_x, value_x):
        assert key_x.shape == query_x.shape
        assert query_x.shape == value_x.shape
        assert self.dimension % self.num_heads == 0

        WX_shape = (key_x.shape[-1], self.dimension)
        WK = self.param("WK", self.WK_init, WX_shape)
        WQ = self.param("WQ", self.WQ_init, WX_shape)
        WV = self.param("WV", self.WV_init, WX_shape)

        if self.mask_queries_from is None:
            mask_from_concrete = key_x.shape[0]
        else:
            assert self.mask_queries_from <= key_x.shape[0]
            mask_from_concrete = self.mask_queries_from

        mask_zeros = jnp.zeros((1, 1, mask_from_concrete))
        mask_neginf = jnp.full((1, 1, key_x.shape[0] - mask_from_concrete), -jnp.inf)
        mask = jnp.concatenate([mask_zeros, mask_neginf], axis=-1)

        K = jnp.dot(key_x, WK)
        Q = jnp.dot(query_x, WQ)
        V = jnp.dot(value_x, WV)

        K_heads = self._split_heads(K)
        Q_heads = self._split_heads(Q)
        V_heads = self._split_heads(V)

        head_logits = (
            jnp.matmul(Q_heads, K_heads.transpose((0, 2, 1)))
            / jnp.sqrt(self.dimension // self.num_heads)
            + mask
        )

        head_weights = nn.softmax(head_logits)
        head_result = jnp.matmul(head_weights, V_heads)
        return head_result.transpose((1, 0, 2)).reshape((-1, self.dimension))
