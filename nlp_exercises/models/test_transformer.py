from tkinter import W
import numpy as np
from jax import numpy as jnp
from jax import random, vmap
from models.transformer import TransformerDecoderBlock, TransformerEncoderBlock, TransformerEncoderDecoder

def test_encoder_smoke_single_ex():
    dummy_inputs = np.ones((100, 512))
    encoder = TransformerEncoderBlock(dimension=512, dimension_inner=2048, num_heads=8)

    random_key = random.PRNGKey(0)
    params = encoder.init(random_key, dummy_inputs)
    outs = encoder.apply(params, dummy_inputs)

    assert outs.shape == (100, 512)

def test_decoder_smoke_single_ex():
    dummy_inputs = np.ones((100, 512))
    dummy_encoder_outputs = np.ones((100, 512))
    decoder = TransformerDecoderBlock(dimension=512, dimension_inner=2048, num_heads=8)
    random_key = random.PRNGKey(0)
    params = decoder.init(random_key, dummy_inputs, dummy_encoder_outputs)
    outs = decoder.apply(params, dummy_inputs, dummy_encoder_outputs)

    assert outs.shape == (100, 512)

def test_transformer_encoder_decoder_smoke_single_ex():
    dummy_tokens = np.ones((16, 100), dtype=jnp.int32)
    transformer = TransformerEncoderDecoder(
        dimension=128,
        num_heads=4,
        dimension_inner=512,
        num_blocks=6,
        num_embeddings=1024
    )

    random_key = random.PRNGKey(0)
    params = transformer.init(random_key, dummy_tokens)
    outs = transformer.apply(params, dummy_tokens)

    assert outs.shape == (16, 100, 1024)



    
    