from layers.attention import Attention
import numpy as np
from jax import random
from jax import numpy as jnp
from flax import linen as nn

def test_unmasked_attention_trivial():
    keys_x = np.array([[1, 2], [-5, -10]])
    queries_x = np.array([[3, 4], [5, 6]])
    values_x = np.array([[7, 8], [9, 10]])

    attention = Attention(dimension=1, 
                          WK_init=lambda _, __: np.array([[1], [0]]),
                          WQ_init=lambda _, __: np.array([[0], [1]]),
                          WV_init=lambda _, __: np.array([[1], [1]]))


    random_key = random.PRNGKey(0)
    params = attention.init(random_key, keys_x, queries_x, values_x)
    result = attention.apply(params, keys_x, queries_x, values_x)

    QKT = np.array([[4, -20], [6, -30]])
    weights1 = nn.softmax(QKT[0])
    weights2 = nn.softmax(QKT[1])

    V = np.array([[15], [19]])
    assert weights1.dot(V) == result[0]
    assert weights2.dot(V) == result[1]

def test_masked_attention_trivial():
    keys_x = np.array([[1, 2], [-5, -10]])
    queries_x = np.array([[3, 4], [5, 6]])
    values_x = np.array([[7, 8], [9, 10]])

    attention = Attention(dimension=1, mask_queries_from=1, 
                          WK_init=lambda _, __: np.array([[1], [0]]),
                          WQ_init=lambda _, __: np.array([[0], [1]]),
                          WV_init=lambda _, __: np.array([[1], [1]]))


    random_key = random.PRNGKey(0)
    params = attention.init(random_key, keys_x, queries_x, values_x)
    result = attention.apply(params, keys_x, queries_x, values_x)

    QKT = np.array([[4, -np.inf], [6, -np.inf]])
    weights1 = nn.softmax(QKT[0])
    weights2 = nn.softmax(QKT[1])

    V = np.array([[15], [19]])
    assert weights1.dot(V) == result[0]
    assert weights2.dot(V) == result[1]
    