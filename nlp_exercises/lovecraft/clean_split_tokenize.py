import random
import re
import spacy
import sentencepiece as spm
import gin
import numpy as np
from pathlib import Path


@gin.configurable
def clean_split_tokenize(
    random_seed: int, vocab_size: int, train_portion: float, unk_id: int, pad_id: int
):
    for child in ["train", "val"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    with (input_dir / "all_novels.txt").open("r") as f:
        text = f.read()

    text = text.replace(" . . . .", "...")
    text = text.replace(" . . .", "...")
    text = text.replace("\n\n\n", "")
    text = text.replace("Return to Table of Contents \n", "")

    novel_names = re.findall(r"(.+)\n{1,2}\(\d\d\d\d\) *\n\n", text)
    novel_names = [n.strip(" ") for n in novel_names]

    novels = re.split(r".+\n{1,2}\(\d\d\d\d\) *\n\n", text)[1:]
    indices = list(range(len(novels)))
    random.seed(random_seed)
    random.shuffle(indices)
    train_novels = round(len(novels) * train_portion)
    train_indices = indices[:train_novels]
    val_indices = indices[train_novels:]

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 3 * 10**6
    for component in list(nlp.component_names):
        nlp.remove_pipe(component)
    nlp.add_pipe("sentencizer")

    train_text = "".join(novels[i] for i in train_indices)
    train_doc = nlp(train_text)

    with (output_dir / "train_sentence_list.txt").open("w") as f:
        newline = "\n"
        f.writelines(f"{s.text.replace(newline, '')}\n" for s in train_doc.sents)

    spm.SentencePieceTrainer.train(
        input=output_dir / "train_sentence_list.txt",
        model_prefix=output_dir / f"sentencepiece_{vocab_size}",
        vocab_size=vocab_size,
        bos_id=-1,
        eos_id=-1,
        unk_id=unk_id,
        pad_id=pad_id,
    )

    sp = spm.SentencePieceProcessor(
        model_file=str(output_dir / f"sentencepiece_{vocab_size}.model")
    )
    encoded = [sp.encode(novel, out_type=int) for novel in novels]

    for i in train_indices:
        name = novel_names[i]
        encoded_novel = encoded[i]
        np.save(
            output_dir / f"train_encoded_{name.replace(' ', '_')}.npy", encoded_novel
        )

    for i in val_indices:
        name = novel_names[i]
        encoded_novel = encoded[i]
        np.save(output_dir / f"val_encoded_{name.replace(' ', '_')}.npy", encoded_novel)


if __name__ == "__main__":
    input_dir = Path("data/lovecraft/input")
    output_dir = Path("data/lovecraft/dataset")

    gin.parse_config_file(input_dir / "config.gin")
    clean_split_tokenize()
