import functools
from flax import linen as nn
from jax import numpy as jnp
import jax

from layers.attention import MultiHeadAttention


class PositionWiseFFN(nn.Module):
    dimension: int
    dtype: jnp.dtype = jnp.float16

    @nn.compact
    def __call__(self, inputs):
        linear1 = nn.linear.Dense(
            features=self.dimension,
            param_dtype=self.dtype,
            kernel_init=nn.initializers.normal(0.02),
        )
        linear2 = nn.linear.Dense(
            features=inputs.shape[-1],
            param_dtype=self.dtype,
            kernel_init=nn.initializers.normal(0.02),
        )

        out = jax.nn.gelu(linear1(inputs))
        out = linear2(out)

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


class TransformerDecoderWithContextBlock(nn.Module):
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


class TransformerDecoderWithoutContextBlock(nn.Module):
    dimension: int
    dimension_inner: int
    num_heads: int
    dropout_rate: float
    dtype: jnp.dtype = jnp.float16

    @nn.compact
    def __call__(self, inputs, train):
        assert inputs.shape[-1] == self.dimension

        self_attention = MultiHeadAttention(
            dimension=self.dimension,
            num_heads=self.num_heads,
            masked=True,
            dtype=self.dtype,
        )
        layer_norms = [nn.normalization.LayerNorm() for _ in range(2)]
        dropouts = [
            nn.stochastic.Dropout(rate=self.dropout_rate, deterministic=not train)
            for _ in range(2)
        ]
        position_wise = PositionWiseFFN(
            dimension=self.dimension_inner, dtype=self.dtype
        )

        out = layer_norms[0](
            inputs + dropouts[0](self_attention(inputs, inputs, inputs))
        )
        out = layer_norms[1](out + dropouts[1](position_wise(out)))

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
            TransformerDecoderWithContextBlock(
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

        logits = embed.attend(out)
        return logits


@functools.partial(
    nn.vmap,
    variable_axes={"params": None, "dropout": None},
    split_rngs={"params": False, "dropout": False},
    in_axes=(0, None),
    out_axes=0,
    axis_name="batch",
)
class TransformerDecoder(nn.Module):
    dimension: int
    num_heads: int
    dimension_inner: int
    num_blocks: int
    num_embeddings: int
    dropout_rate: float
    dtype: jnp.dtype = jnp.float16

    @nn.compact
    def __call__(self, inputs, train: bool):
        embed = nn.linear.Embed(
            num_embeddings=self.num_embeddings,
            features=self.dimension,
            param_dtype=self.dtype,
        )
        input_dropout = nn.stochastic.Dropout(
            rate=self.dropout_rate, deterministic=not train
        )
        position_vectors = self.param(
            "position_vectors",
            nn.initializers.normal(0.02, dtype=self.dtype),
            (inputs.shape[-1], self.dimension),
        )
        decoder_stack = [
            TransformerDecoderWithoutContextBlock(
                dimension=self.dimension,
                dimension_inner=self.dimension_inner,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )
            for _ in range(self.num_blocks)
        ]

        embedded_inputs = embed(inputs)
        out = embedded_inputs + position_vectors
        out = input_dropout(out)

        for decoder_block in decoder_stack:
            out = decoder_block(out, train)

        logits = embed.attend(out)
        return logits
