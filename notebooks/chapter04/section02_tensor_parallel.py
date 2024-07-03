from pathlib import Path

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_from_scratch.transformer.transformer import Transformer

from .utils import (
    create_padding_mask,
    create_subsequent_mask,
    load_dataset,
)

tp_mesh = init_device_mesh("cuda", (8,))
rank = tp_mesh.get_rank()


def train(model, batch_size, n_epochs, train_dataset, dataset_info, device):
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
    ).to(device)

    # レイヤーごとに並列化の方法を定義
    for module in [model.encoder, model.decoder]:
        # エンコーダ・デコーダのブロック
        for block in module.blocks:
            tp_plan_block = {
                "attention.linear_o": RowwiseParallel(),
                "feed_forward.0": ColwiseParallel(),
                "feed_forward.2": RowwiseParallel(),
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=tp_plan_block,
            )
            # マルチヘッドアテンションのヘッドごと
            for attention in block.attention.heads:
                tp_plan_attention = {
                    "linear_q": ColwiseParallel(),
                    "linear_k": ColwiseParallel(),
                    "linear_v": ColwiseParallel(),
                }
                parallelize_module(
                    module=attention,
                    device_mesh=tp_mesh,
                    parallelize_plan=tp_plan_attention,
                )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=dataset_info["collate_fn"],
    )

    PAD_ID = dataset_info["vocab_src"]["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)  # クロスエントロピー
    lr = 0.0001  # 学習率
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.95)
    pbar = tqdm(total=n_epochs, disable=rank != 0)
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
                src_texts.to(device),
                tgt_input.to(device),
                tgt_output.to(device),
            )
            src_padding_mask, tgt_mask = (
                src_padding_mask.to(device),
                tgt_mask.to(device),
            )

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
    device = "cuda"
    data_dir = Path("small_parallel_enja")
    train_dataset, dataset_info = load_dataset(data_dir)

    train(
        batch_size=64,
        n_epochs=10,
        train_dataset=train_dataset,
        dataset_info=dataset_info,
        device=device,
    )
    if rank == 0:
        print(torch.cuda.max_memory_allocated())


if __name__ == "__main__":
    main()
