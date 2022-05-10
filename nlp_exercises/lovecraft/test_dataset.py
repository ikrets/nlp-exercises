import numpy as np
from lovecraft.dataset import Dataset
import itertools


def test_tokens_to_example():
    d = Dataset([], [], batch_size=2, token_count=3, pad_id=0, vocab_size=15)

    input_tokens, one_hot = d._tokens_to_example(np.array([[1, 2, 3]]))
    np.testing.assert_array_equal(input_tokens, [[0, 1, 2]])
    np.testing.assert_array_equal(np.argmax(one_hot, axis=-1), [[1, 2, 3]])


def test_train_iterator():
    train_novels = np.array([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]])
    val_novels = np.array([])

    d = Dataset(
        train_novels, val_novels, batch_size=2, token_count=3, pad_id=0, vocab_size=15
    )
    batches = list(itertools.islice(d.train_iterator(), 2))

    for batch in batches:
        assert batch["tokens"].shape == (2, 3)
        assert batch["one_hot"].shape == (2, 3, 15)

        for i in range(batch["tokens"].shape[0]):
            assert batch["tokens"][i, 0] == 0
            np.testing.assert_array_equal(
                np.argmax(batch["one_hot"][i], axis=-1)[:2], batch["tokens"][i, 1:]
            )
