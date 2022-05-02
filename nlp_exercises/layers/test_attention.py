from layers.attention import MultiHeadAttention
import numpy as np
from jax import random
from jax import numpy as jnp
from flax import linen as nn


def test_unmasked_attention_one_head():
    keys_x = np.array([[1, 2], [-5, -10]])
    queries_x = np.array([[3, 4], [5, 6]])
    values_x = np.array([[7, 8], [9, 10]])

    attention = MultiHeadAttention(
        dimension=1,
        num_heads=1,
        WK_init=lambda _, __: np.array([[1], [0]]),
        WQ_init=lambda _, __: np.array([[0], [1]]),
        WV_init=lambda _, __: np.array([[1], [1]]),
    )

    random_key = random.PRNGKey(0)
    params = attention.init(random_key, keys_x, queries_x, values_x)
    result = attention.apply(params, keys_x, queries_x, values_x)

    QKT = np.array([[4, -20], [6, -30]])
    weights1 = nn.softmax(QKT[0])
    weights2 = nn.softmax(QKT[1])

    V = np.array([[15], [19]])
    assert weights1.dot(V) == result[0]
    assert weights2.dot(V) == result[1]


def test_masked_attention_one_head():
    keys_x = np.array([[1, 2], [-5, -10]])
    queries_x = np.array([[3, 4], [5, 6]])
    values_x = np.array([[7, 8], [9, 10]])

    attention = MultiHeadAttention(
        dimension=1,
        masked=True,
        WK_init=lambda _, __: np.array([[1], [0]]),
        WQ_init=lambda _, __: np.array([[0], [1]]),
        WV_init=lambda _, __: np.array([[1], [1]]),
    )

    random_key = random.PRNGKey(0)
    params = attention.init(random_key, keys_x, queries_x, values_x)
    result = attention.apply(params, keys_x, queries_x, values_x)

    QKT = np.array([[4, -np.inf], [6, -30]])
    weights1 = nn.softmax(QKT[0])
    weights2 = nn.softmax(QKT[1])

    V = np.array([[15], [19]])
    assert weights1.dot(V) == result[0]
    assert weights2.dot(V) == result[1]


def test_unmasked_attention_multiple_heads():
    x = np.array([[1, 2], [-5, -10], [-2, -4]])

    W_init = lambda _, __: np.array([[1, 0, 1, -1], [0, 1, 1, -1]])

    attention = MultiHeadAttention(
        dimension=4, num_heads=4, WK_init=W_init, WQ_init=W_init, WV_init=W_init
    )

    random_key = random.PRNGKey(0)
    params = attention.init(random_key, x, x, x)
    result = attention.apply(params, x, x, x)

    V = np.array([[1, 2, 3, -3], [-5, -10, -15, 15], [-2, -4, -6, 6]])

    for i in range(4):
        QKTi = jnp.matmul(V[:, i : i + 1], (V[:, i : i + 1].T))
        weightsi = nn.softmax(QKTi)
        assert jnp.array_equal(weightsi.dot(V[:, i]), result[:, i])
