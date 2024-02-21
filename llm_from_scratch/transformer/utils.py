import math

import torch
from torch import Tensor, nn


def layer_norm(x: Tensor, eps=1e-6):
    """Layer normalization.
    Args:
        x (Tensor): Input tensor.
        eps (float): Epsilon.
    Returns:
        Tensor: Output tensor.
    """
    return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True, unbiased=False).add(
        eps
    )


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps=1e-6):
        """Layer 正規化.
        Args:
            d_model (int): 埋め込み次元数
            eps (float): ゼロ除算防止のための微小値
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def sinusoidal_position_encoding(d_model: int, sequence_length: int) -> Tensor:
    pe = torch.zeros(sequence_length, d_model)
    dim_even = torch.arange(0, d_model, 2)
    dim_odd = torch.arange(1, d_model, 2)
    # unsqueeze で (sequence_length, 1) の形にする
    pos = torch.arange(0, sequence_length).unsqueeze(1)

    pe[:, dim_even] = torch.sin(pos / (10000 ** (dim_even / d_model)))
    pe[:, dim_odd] = torch.cos(pos / (10000 ** ((dim_odd - 1) / d_model)))

    # unsqueeze(0) でミニバッチの次元を先頭に追加
    return pe.unsqueeze(0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_sequence_length: int):
        super().__init__()
        pe = sinusoidal_position_encoding(d_model, max_sequence_length)
        # 値を最適化しないので register_buffer で登録
        self.scale = math.sqrt(d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        sequence_length = x.size(1)
        x = x * self.scale + self.pe[:, :sequence_length]
        return x
