import torch
from torch import Tensor, nn

from llm_from_scratch.transformer.attention import MultiHeadAttention
from llm_from_scratch.transformer.utils import LayerNorm, PositionalEncoding


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        """デコーダーブロック.

        Args:
            d_model (int): モデルの埋め込み次元数
            n_heads (int): ヘッド数
            d_k (int): クエリ, キーの次元数
            d_v (int): バリューの次元数
            d_ff (int): フィードフォワードネットワークの隠れ層の次元数
        """
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_k, d_v, d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELUを使用
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """デコーダーブロックの順伝播.

        Args:
            x (Tensor): 入力テンソル. shapeは(batch_size, seq_len, d_model).
            mask (Tensor, optional): マスク. shapeは(batch_size, seq_len, seq_len).

        Returns:
            Tensor: 出力テンソル. shapeは(batch_size, seq_len, d_model).
        """
        norm_x = self.layer_norm1(x)
        x_attention = self.attention(norm_x, norm_x, norm_x, mask=mask)
        x = x + x_attention  # 残差接続
        
        norm_x = self.layer_norm2(x)
        x_ff = self.feed_forward(norm_x)
        x = x + x_ff  # 残差接続
        return x


class DecoderOnlyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_len: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
    ):
        """デコーダーのみのモデル.

        Args:
            vocab_size (int): 語彙サイズ
            max_sequence_len (int): 最大シーケンス長
            d_model (int): モデルの埋め込み次元数
            n_blocks (int): ブロック数
            n_heads (int): ヘッド数
            d_k (int): クエリ, キーの次元数
            d_v (int): バリューの次元数
            d_ff (int): フィードフォワードネットワークの隠れ層の次元数
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_sequence_len)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_blocks)]
        )
        self.layer_norm_final = LayerNorm(d_model)  # 最終レイヤー正規化
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """デコーダーのみのモデルの順伝播.

        Args:
            x (Tensor): 入力テンソル. shapeは(batch_size, seq_len).
            mask (Tensor, optional): マスク. shapeは(batch_size, seq_len, seq_len).

        Returns:
            Tensor: 出力テンソル. shapeは(batch_size, seq_len, vocab_size).
        """
        x = self.embedding(x)
        x = self.pe(x)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.layer_norm_final(x)
        logits = self.linear(x)
        return logits

    @torch.inference_mode()
    def generate(
        self, 
        input_ids: Tensor, 
        max_length: int = 100, 
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        do_sample: bool = True,
        eos_token_id: int | None = None
    ) -> Tensor:
        """デコーダーのみのモデルを使用してテキストを生成する.

        Args:
            input_ids (Tensor): 開始トークンID. shapeは(batch_size, seq_len).
            max_length (int): 生成するシーケンスの最大長.
            temperature (float): サンプリングの温度.
            top_k (int): top-kサンプリングのk値. 0より大きい場合、最も確率の高いk個のトークンからサンプリング.
            top_p (float): top-pサンプリングのp値. 0より大きい場合、累積確率がtop_p以上のトークンからサンプリング.
            do_sample (bool): サンプリングを行うかどうか. Falseの場合、貪欲法を使用.
            eos_token_id (int, optional): 終了トークンID.

        Returns:
            Tensor: 生成されたトークンID. shapeは(batch_size, <= max_length).
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated = input_ids
        
        seq_len = generated.shape[1]
        mask = torch.triu(torch.ones((seq_len, seq_len), device=device) * float('-inf'), diagonal=1)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        for _ in range(max_length - seq_len):
            logits = self(generated, mask=mask)[:, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
                
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_values)
                
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
            generated = torch.cat([generated, next_token], dim=1)
            
            seq_len = generated.shape[1]
            mask = torch.triu(torch.ones((seq_len, seq_len), device=device) * float('-inf'), diagonal=1)
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            if eos_token_id is not None and (next_token == eos_token_id).any():
                if (next_token == eos_token_id).all():
                    break
        
        return generated


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """因果的マスクを作成する.

    Args:
        seq_len (int): シーケンス長
        device (torch.device): デバイス

    Returns:
        torch.Tensor: 因果的マスク. shapeは(1, seq_len, seq_len).
    """
    mask = torch.triu(torch.ones((seq_len, seq_len), device=device) * float('-inf'), diagonal=1)
    return mask.unsqueeze(0)  # バッチ次元を追加