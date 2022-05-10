import gin
import functools
import numpy as np
import sentencepiece as spm
import jax
import jax.numpy as jnp


@gin.configurable(allowlist=["batch_size", "token_count", "pad_id", "vocab_size"])
class Dataset:
    def __init__(
        self,
        train_novels: list[np.ndarray],
        val_novels: list[np.ndarray],
        batch_size: int,
        token_count: int,
        pad_id: int,
        vocab_size: int,
    ):
        self.train_novels = train_novels
        self.val_novels = val_novels

        self.batch_size = batch_size
        self.token_count = token_count
        self.pad_id = pad_id
        self.vocab_size = vocab_size

    @functools.partial(jax.vmap, in_axes=(None, 0))
    def _tokens_to_example(self, tokens):
        one_hot = jax.nn.one_hot(tokens, self.vocab_size)
        padded_left = jnp.pad(
            tokens,
            (1, 0),
            "constant",
            constant_values=self.pad_id,
        )
        return padded_left[: self.token_count], one_hot

    def _get_padded_tokens(self, novels, story_id, start_position):
        chunk = novels[story_id][start_position : start_position + self.token_count]
        return np.pad(
            chunk,
            (0, self.token_count - len(chunk)),
            "constant",
            constant_values=self.pad_id,
        )

    def train_iterator(self):
        while True:
            story_ids = np.random.randint(0, len(self.train_novels), self.batch_size)
            padded_tokens = []
            for story in story_ids:
                start_position = np.random.randint(0, len(self.train_novels[story]))
                padded_tokens.append(
                    self._get_padded_tokens(self.train_novels, story, start_position)
                )

            batch_inputs, one_hots = self._tokens_to_example(jnp.array(padded_tokens))
            yield {"tokens": batch_inputs, "one_hot": one_hots}

    def _val_iterator_story_ids_and_start_positions(self):
        for i, val_novel in enumerate(self.val_novels):
            for j in range(0, len(val_novel), self.token_count):
                yield i, j

    def val_iterator(self):
        story_position_it = self._val_iterator_story_ids_and_start_positions()
        padded_tokens = []
        while True:
            try:
                next_story, next_position = next(story_position_it)
            except StopIteration:
                return

            padded_tokens.append(
                self._get_padded_tokens(self.val_novels, next_story, next_position)
            )

            if len(padded_tokens) == self.batch_size:
                batch_inputs, one_hots = self._tokens_to_example(
                    jnp.array(padded_tokens)
                )
                yield {
                    "tokens": batch_inputs,
                    "one_hot": one_hots,
                }


def decode_tokens(tokens, vocab_size):
    sp = spm.SentencePieceProcessor(
        model_file=f"data/lovecraft/dataset/sentencepiece_{vocab_size}.model"
    )

    return sp.decode(tokens.tolist())
