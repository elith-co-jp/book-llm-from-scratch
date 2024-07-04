# 実行コマンド例: python section02_tensor_parallel.py
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torchtext import transforms
from torchtext.vocab import build_vocab_from_iterator


def create_padding_mask(pad_id: int, batch_tokens: Tensor):
    # batch_tokens.shape == (batch_size, sequence_length)
    mask = batch_tokens == pad_id
    mask = mask.unsqueeze(1)
    return mask


def create_subsequent_mask(batch_tokens: Tensor):
    sequence_len = batch_tokens.size(1)
    mask = torch.triu(
        torch.full((sequence_len, sequence_len), 1),
        diagonal=1,
    )
    mask = mask == 1
    mask = mask.unsqueeze(0)
    return mask


def iter_corpus(
    path: Path,
    bos: str | None = "<bos>",
    eos: str | None = "<eos>",
) -> Iterator[list[str]]:
    with path.open("r") as f:
        for line in f:
            if bos:
                line = bos + " " + line
            if eos:
                line = line + " " + eos
            yield line.split()


class Collator:
    def __init__(self, src_transforms, tgt_transforms):
        self.src_transforms = src_transforms
        self.tgt_transforms = tgt_transforms

    def __call__(self, batch):
        src_texts, tgt_texts = [], []
        for s, t in batch:
            src_texts.append(s)
            tgt_texts.append(t)

        src_texts = self.src_transforms(src_texts)
        tgt_texts = self.tgt_transforms(tgt_texts)

        return src_texts, tgt_texts


# データの準備
def load_dataset(data_dir):
    train_ja = data_dir / "train.ja.000"
    train_en = data_dir / "train.en.000"
    train_tokens_ja = [tokens for tokens in iter_corpus(train_ja)]
    train_tokens_en = [tokens for tokens in iter_corpus(train_en)]
    vocab_ja = build_vocab_from_iterator(
        iterator=train_tokens_ja,
        specials=("<unk>", "<pad>", "<bos>", "<eos>"),
    )
    vocab_ja.set_default_index(vocab_ja["<unk>"])
    vocab_en = build_vocab_from_iterator(
        iterator=train_tokens_en,
        specials=("<unk>", "<pad>", "<bos>", "<eos>"),
    )
    vocab_en.set_default_index(vocab_en["<unk>"])

    src_transforms = transforms.Sequential(
        transforms.VocabTransform(vocab_ja),
        transforms.ToTensor(padding_value=vocab_ja["<pad>"]),
    )
    tgt_transforms = transforms.Sequential(
        transforms.VocabTransform(vocab_en),
        transforms.ToTensor(padding_value=vocab_en["<pad>"]),
    )
    train_dataset = list(zip(train_tokens_ja, train_tokens_en))
    src_max_len = max(len(tokens) for tokens in train_tokens_ja)
    tgt_max_len = max(len(tokens) for tokens in train_tokens_en)
    max_len = max(src_max_len, tgt_max_len)
    dataset_info = {
        "src_vocab_size": len(vocab_ja),
        "tgt_vocab_size": len(vocab_en),
        "max_sequence_len": max_len,
        "collate_fn": Collator(src_transforms, tgt_transforms),
        "vocab_src": vocab_ja,
        "vocab_tgt": vocab_en,
    }
    return train_dataset, dataset_info
