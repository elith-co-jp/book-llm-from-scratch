# 実行コマンド例: python section02_data_parallel.py
import os
from pathlib import Path
from typing import Iterator

import torch
from torch.distributed import init_process_group
import torch.multiprocessing as mp
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtext import transforms
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from llm_from_scratch.transformer.transformer import Transformer

from .utils import create_padding_mask, create_subsequent_mask, load_dataset


def train(rank, n_gpu, batch_size, n_epochs, train_dataset, dataset_info):
    init_process_group("nccl", rank=rank, world_size=n_gpu)
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
    for _ in range(n_epochs):
        pbar.update(1)
        for src_texts, tgt_texts in train_loader:
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
