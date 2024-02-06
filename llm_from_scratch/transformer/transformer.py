from torch import nn
from torch import Tensor
from llm_from_scratch.transformer.attention import MultiHeadAttention
from llm_from_scratch.transformer.utils import LayerNorm, sinusoidal_position_encoding


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_k, d_v, d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x_attention = self.attention(x, x, x)
        x = self.layer_norm1(x + x_attention)
        x_ff = self.feed_forward(x)
        x = self.layer_norm2(x + x_ff)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model: int, n_blocks: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_blocks: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_k, d_v, d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.attention_souce_target = MultiHeadAttention(n_heads, d_k, d_v, d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm3 = LayerNorm(d_model)

    def forward(self, x: Tensor, encoder_output: Tensor, mask: Tensor|None = None) -> Tensor:
        x_attention = self.attention(x, x, x, mask=mask)
        x = self.layer_norm1(x + x_attention)
        x_attention_souce_target = self.attention_souce_target(x, encoder_output, encoder_output)
        x = self.layer_norm2(x + x_attention_souce_target)
        x_ff = self.feed_forward(x)
        x = self.layer_norm3(x + x_ff)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model: int, n_blocks: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_blocks)])

    def forward(self, x: Tensor, encoder_output: Tensor, mask: Tensor|None = Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, encoder_output, mask=mask)
        return x
