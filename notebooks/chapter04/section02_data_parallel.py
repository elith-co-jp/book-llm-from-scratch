import torch
from pathlib import Path
from torch import Tensor
from typing import Iterator
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torchtext import transforms

import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from llm_from_scratch.transformer.transformer import Transformer


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


def create_collate_fn(src_transforms, tgt_transforms):
    def collate_fn(batch: Tensor) -> tuple[Tensor, Tensor]:
        src_texts, tgt_texts = [], []
        for s, t in batch:
            src_texts.append(s)
            tgt_texts.append(t)

        src_texts = src_transforms(src_texts)
        tgt_texts = tgt_transforms(tgt_texts)

        return src_texts, tgt_texts

    return collate_fn


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
        "collate_fn": create_collate_fn(src_transforms, tgt_transforms),
        "vocab_src": vocab_ja,
        "vocab_tgt": vocab_en,
    }
    train_dataset, dataset_info


def train(rank, n_gpu, batch_size, n_epochs, train_dataset, dataset_info):
    dist.init_process_group("gloo", rank=rank, world_size=n_gpu)
    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    # create local model
    embedding_dim = 512
    n_blocks = 6
    n_heads = 8
    expansion_rate = 1

    # 最も長い文章の長さを取得
    model = Transformer(
        dataset_info["src_vocab_size"],
        dataset_info["tgt_vocab_size"],
        max_sequence_len=dataset_info["max_sequence_len"],
        d_model=embedding_dim,
        n_blocks=n_blocks,
        n_heads=n_heads,
        d_k=embedding_dim,
        d_v=embedding_dim,
        d_ff=embedding_dim * expansion_rate,
    ).to(rank)
    # ここで各 rank の GPU にモデルを配置
    model = DDP(model, device_ids=[rank])

    # rank ごとにデータを分割するためのサンプラーを作成
    sampler = DistributedSampler(
        train_dataset, num_replicas=n_gpu, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=dataset_info["collate_fn"],
    )

    PAD_ID = dataset_info["vocab_src"]["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)  # クロスエントロピー
    lr = 0.0001  # 学習率
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.95)
    pbar = tqdm(total=n_epochs, desc=f"rank{rank}", position=rank)
    for epoch in range(n_epochs):
        pbar.update(1)
        for i, (src_texts, tgt_texts) in enumerate(train_loader):
            # tgt の入力は最後の単語を除く
            tgt_input = tgt_texts[:, :-1]
            # tgt の出力は最初の単語を除く
            tgt_output = tgt_texts[:, 1:]
            src_padding_mask = create_padding_mask(PAD_ID, src_texts)
            tgt_padding_mask = create_padding_mask(PAD_ID, tgt_input)
            tgt_subsequent_mask = create_subsequent_mask(tgt_input)
            tgt_mask = tgt_padding_mask + tgt_subsequent_mask
            # Tensor のデバイスを設定
            src_texts, tgt_input, tgt_output = (
                src_texts.to(rank),
                tgt_input.to(rank),
                tgt_output.to(rank),
            )
            src_padding_mask, tgt_mask = src_padding_mask.to(rank), tgt_mask.to(rank)

            # モデル出力を取得
            out = model(
                src_texts, tgt_input, src_padding_mask, tgt_mask, src_padding_mask
            )
            # 出力と教師データを1次元に変換
            out_flat = out.view(-1, out.size(-1))
            tgt_flat = tgt_output.flatten()
            # 誤差関数を計算
            loss = criterion(out_flat, tgt_flat)
            optimizer.zero_grad()
            # 誤差逆伝播
            loss.backward()
            optimizer.step()
        scheduler.step()

    if rank == 0:
        torch.save(model.state_dict(), "transformer.pth")


def main():
    n_gpu = 4
    batch_size = 64
    n_epochs = 10
    data_dir = Path("small_parallel_enja")
    train_dataset, dataset_info = load_dataset(data_dir)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    mp.spawn(
        train,
        args=(n_gpu, batch_size, n_epochs, train_dataset, dataset_info),
        nprocs=n_gpu,
        join=True,
    )


if __name__ == "__main__":
    main()
