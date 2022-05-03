import functools
from base64 import encode
from flax import linen as nn
from jax import numpy as jnp

from layers.attention import MultiHeadAttention


class PositionWiseFFN(nn.Module):
    dimension: int

    @nn.compact
    def __call__(self, inputs):
        linear1 = nn.linear.Dense(features=self.dimension)
        linear2 = nn.linear.Dense(features=inputs.shape[-1])

        out = jnp.maximum(0, linear1(inputs))
        out = jnp.maximum(0, linear2(out))

        return out


class TransformerEncoderBlock(nn.Module):
    dimension: int
    dimension_inner: int
    num_heads: int

    @nn.compact
    def __call__(self, inputs):
        assert inputs.shape[-1] == self.dimension

        attention = MultiHeadAttention(
            dimension=self.dimension, num_heads=self.num_heads
        )
        layer_norms = [nn.normalization.LayerNorm() for _ in range(2)]
        position_wise = PositionWiseFFN(dimension=self.dimension_inner)

        out = layer_norms[0](inputs + attention(inputs, inputs, inputs))
        out = layer_norms[1](out + position_wise(out))

        return out


class TransformerDecoderBlock(nn.Module):
    dimension: int
    dimension_inner: int
    num_heads: int

    @nn.compact
    def __call__(self, decoder_inputs, encoder_outputs):
        assert decoder_inputs.shape[-1] == self.dimension
        assert encoder_outputs.shape[-1] == self.dimension

        self_attention = MultiHeadAttention(
            dimension=self.dimension, num_heads=self.num_heads, masked=True
        )
        cross_attention = MultiHeadAttention(
            dimension=self.dimension, num_heads=self.num_heads
        )
        layer_norms = [nn.normalization.LayerNorm() for _ in range(3)]
        position_wise = PositionWiseFFN(dimension=self.dimension_inner)

        out = layer_norms[0](
            decoder_inputs
            + self_attention(decoder_inputs, decoder_inputs, decoder_inputs)
        )
        out = layer_norms[1](
            out
            + cross_attention(
                query_x=out, key_x=encoder_outputs, value_x=encoder_outputs
            )
        )
        out = layer_norms[2](out + position_wise(out))

        return out


@functools.partial(
    nn.vmap,
    variable_axes={"params": None},
    split_rngs={"params": False},
    in_axes=0,
    out_axes=0,
    axis_name="batch",
)
class TransformerEncoderDecoder(nn.Module):
    dimension: int
    num_heads: int
    dimension_inner: int
    num_blocks: int
    num_embeddings: int

    @nn.compact
    def __call__(self, inputs):
        embed = nn.linear.Embed(
            num_embeddings=self.num_embeddings, features=self.dimension
        )
        position_vectors = self.param(
            "position_vectors",
            nn.initializers.lecun_normal(),
            (inputs.shape[-1], self.dimension),
        )
        encoder_stack = [
            TransformerEncoderBlock(
                dimension=self.dimension,
                dimension_inner=self.dimension_inner,
                num_heads=self.num_heads,
            )
            for _ in range(self.num_blocks)
        ]
        decoder_stack = [
            TransformerDecoderBlock(
                dimension=self.dimension,
                dimension_inner=self.dimension_inner,
                num_heads=self.num_heads,
            )
            for _ in range(self.num_blocks)
        ]

        embedded_inputs = embed(inputs)
        out = embedded_inputs + position_vectors

        for encoder_block in encoder_stack:
            out = encoder_block(out)

        encoder_outputs = out

        for decoder_block in decoder_stack:
            out = decoder_block(out, encoder_outputs=encoder_outputs)

        projected_outputs = embed.attend(out)
        return nn.softmax(projected_outputs)
