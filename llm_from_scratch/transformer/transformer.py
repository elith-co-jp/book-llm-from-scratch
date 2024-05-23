import torch
from torch import Tensor, nn

from llm_from_scratch.transformer.attention import MultiHeadAttention
from llm_from_scratch.transformer.utils import LayerNorm, PositionalEncoding


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

    def forward(self, x: Tensor, src_padding_mask: Tensor | None = None) -> Tensor:
        x_attention = self.attention(x, x, x, mask=src_padding_mask)
        x = self.layer_norm1(x + x_attention)
        x_ff = self.feed_forward(x)
        x = self.layer_norm2(x + x_ff)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        max_sequence_len: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.pe = PositionalEncoding(d_model, max_sequence_len)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_blocks)]
        )

    def forward(self, x: Tensor, src_padding_mask: Tensor | None = None) -> Tensor:
        """Forward.

        Args:
            x (Tensor): Input tensor. shapeは(batch_size, sequence_length, 1).

        Returns:
            Tensor: Output tensor. shapeは(batch_size, sequence_length, d_model).
        """
        x = self.embedding(x)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x, src_padding_mask=src_padding_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_k, d_v, d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.attention_source_target = MultiHeadAttention(n_heads, d_k, d_v, d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm3 = LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        tgt_mask: Tensor | None = None,
        src_tgt_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x_attention = self.attention(x, x, x, mask=tgt_mask)
        x = self.layer_norm1(x + x_attention)
        x_attention_source_target = self.attention_source_target(
            x, encoder_output, encoder_output, mask=src_tgt_padding_mask
        )
        x = self.layer_norm2(x + x_attention_source_target)
        x_ff = self.feed_forward(x)
        x = self.layer_norm3(x + x_ff)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        max_sequence_len: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.pe = PositionalEncoding(d_model, max_sequence_len)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_blocks)]
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        mask: Tensor | None = None,
        src_tgt_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embedding(x)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x, encoder_output, mask=mask, src_tgt_mask=src_tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_sequence_len: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, max_sequence_len, d_model, n_blocks, n_heads, d_k, d_v, d_ff
        )
        self.decoder = Decoder(
            tgt_vocab_size, max_sequence_len, d_model, n_blocks, n_heads, d_k, d_v, d_ff
        )
        self.linear = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        src_tgt_mask: Tensor | None = None,
    ) -> Tensor:
        encoder_output = self.encoder(src, mask=src_mask)
        decoder_output = self.decoder(
            tgt, encoder_output, mask=tgt_mask, src_tgt_mask=src_tgt_mask
        )
        output = self.linear(decoder_output)
        return output

    @torch.inference_mode
    def inference(self, src: Tensor, bos_token: int) -> Tensor:
        tgt_tokens = torch.tensor([[bos_token]]).to(src.device)

        encoder_output = self.encoder(src)
        for _ in range(20):
            decoder_output = self.decoder(tgt_tokens, encoder_output)
            pred = self.linear(decoder_output)
            pred = torch.tensor([[pred[0, -1].argmax().item()]]).to(src.device)
            tgt_tokens = torch.cat((tgt_tokens, pred), axis=-1)
            if pred[0, 0].item() == 3:
                break

        return tgt_tokens
