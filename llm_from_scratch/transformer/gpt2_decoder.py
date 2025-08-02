import torch
from torch import Tensor, nn

from llm_from_scratch.transformer.attention import MultiHeadAttention
from llm_from_scratch.transformer.utils import LayerNorm, PositionalEncoding


def create_causal_mask(seq_len: int) -> Tensor:
    """因果マスクを作成する（未来のトークンを見えないようにする）
    
    Args:
        seq_len (int): シーケンス長
        
    Returns:
        Tensor: 因果マスク (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


class GPT2Block(nn.Module):
    """GPT-2スタイルのTransformerブロック（decoder-only）"""
    
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.layer_norm1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(n_heads, d_k, d_v, d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT-2ではGELUを使用
            nn.Linear(d_ff, d_model),
        )
        
    def forward(self, x: Tensor, causal_mask: Tensor | None = None) -> Tensor:
        x_norm = self.layer_norm1(x)
        x_attention = self.attention(x_norm, x_norm, x_norm, mask=causal_mask)
        x = x + x_attention  # 残差接続
        
        x_norm = self.layer_norm2(x)
        x_ff = self.feed_forward(x_norm)
        x = x + x_ff  # 残差接続
        
        return x


class GPT2Decoder(nn.Module):
    """GPT-2スタイルのDecoder-onlyモデル"""
    
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
        self.d_model = d_model
        self.max_sequence_len = max_sequence_len
        
        self.token_embedding = nn.Embedding(vocabulary_size, d_model)
        
        self.pe = PositionalEncoding(d_model, max_sequence_len)
        
        self.blocks = nn.ModuleList([
            GPT2Block(d_model, n_heads, d_k, d_v, d_ff) 
            for _ in range(n_blocks)
        ])
        
        self.layer_norm_final = LayerNorm(d_model)
        
        self.lm_head = nn.Linear(d_model, vocabulary_size, bias=False)
        
        self.lm_head.weight = self.token_embedding.weight
        
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids (Tensor): 入力トークンID (batch_size, seq_len)
            
        Returns:
            Tensor: 各位置での次トークンの予測確率 (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        x = self.token_embedding(input_ids)
        x = self.pe(x)
        
        causal_mask = create_causal_mask(seq_len).to(input_ids.device)
        
        for block in self.blocks:
            x = block(x, causal_mask=causal_mask)
            
        x = self.layer_norm_final(x)
        
        logits = self.lm_head(x)
        
        return logits
    
    @torch.inference_mode()
    def generate(
        self, 
        input_ids: Tensor, 
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        pad_token_id: int = 0,
        eos_token_id: int | None = None
    ) -> Tensor:
        """テキスト生成（自己回帰的生成）
        
        Args:
            input_ids (Tensor): 初期入力トークン (batch_size, seq_len)
            max_new_tokens (int): 生成する最大トークン数
            temperature (float): サンプリング温度
            top_k (int): Top-kサンプリング
            top_p (float): Top-pサンプリング（nucleus sampling）
            pad_token_id (int): パディングトークンID
            eos_token_id (int): 終了トークンID
            
        Returns:
            Tensor: 生成されたトークン列 (batch_size, seq_len + max_new_tokens)
        """
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.max_sequence_len:
                generated = generated[:, 1:]
            
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :]  # 最後の位置のlogits
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return generated
    
    @torch.inference_mode()
    def greedy_generate(
        self, 
        input_ids: Tensor, 
        max_new_tokens: int = 50,
        eos_token_id: int | None = None
    ) -> Tensor:
        """貪欲法による生成（最も確率の高いトークンを選択）
        
        Args:
            input_ids (Tensor): 初期入力トークン (batch_size, seq_len)
            max_new_tokens (int): 生成する最大トークン数
            eos_token_id (int): 終了トークンID
            
        Returns:
            Tensor: 生成されたトークン列
        """
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.max_sequence_len:
                generated = generated[:, 1:]
            
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return generated


if __name__ == "__main__":
    vocab_size = 1000
    max_seq_len = 512
    d_model = 256
    n_blocks = 6
    n_heads = 8
    d_k = d_model // n_heads
    d_v = d_k
    d_ff = d_model * 4
    
    model = GPT2Decoder(
        vocabulary_size=vocab_size,
        max_sequence_len=max_seq_len,
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
    )
    
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    generated = model.greedy_generate(input_ids, max_new_tokens=5)
    print(f"Generated shape: {generated.shape}")