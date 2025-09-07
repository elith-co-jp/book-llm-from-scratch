"""GPT model architecture implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        
        # Linear transformations for Q, K, V
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence, embedding
        
        # Calculate Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Attention scores with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask (prevent attending to future tokens)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model."""
    
    def __init__(self, vocab_size, n_embd=768, n_layer=12, n_head=12, 
                 block_size=1024, dropout=0.1):
        super().__init__()
        
        self.block_size = block_size
        self.n_embd = n_embd
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, dropout) 
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # Forward through transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text autoregressively.
        
        Args:
            idx: Initial token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        """
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


class GPTConfig:
    """Configuration for GPT model."""
    
    def __init__(self, **kwargs):
        # Model configuration
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.n_embd = kwargs.get('n_embd', 768)
        self.n_layer = kwargs.get('n_layer', 12)
        self.n_head = kwargs.get('n_head', 12)
        self.block_size = kwargs.get('block_size', 1024)
        self.dropout = kwargs.get('dropout', 0.1)
        
        # Training configuration
        self.batch_size = kwargs.get('batch_size', 8)
        self.learning_rate = kwargs.get('learning_rate', 6e-4)
        self.weight_decay = kwargs.get('weight_decay', 0.1)
        self.max_steps = kwargs.get('max_steps', 100000)
        self.warmup_steps = kwargs.get('warmup_steps', 2000)
        
        # Optimization configuration
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 4)
        self.grad_clip = kwargs.get('grad_clip', 1.0)
        self.use_amp = kwargs.get('use_amp', True)
        
        # Evaluation configuration
        self.eval_interval = kwargs.get('eval_interval', 500)
        self.eval_steps = kwargs.get('eval_steps', 50)
        self.save_interval = kwargs.get('save_interval', 5000)
        
    def get_model_size(self):
        """Calculate model parameter count."""
        # Embedding layers
        params = self.vocab_size * self.n_embd * 2  # token + position
        
        # Transformer blocks
        params_per_block = (
            # Attention
            self.n_embd * 3 * self.n_embd +  # QKV
            self.n_embd * self.n_embd +       # output projection
            # MLP
            self.n_embd * 4 * self.n_embd +   # fc1
            4 * self.n_embd * self.n_embd +   # fc2
            # LayerNorm
            self.n_embd * 4                   # 2 LayerNorms
        )
        params += params_per_block * self.n_layer
        
        # Output layer
        params += self.n_embd * self.vocab_size
        
        return params / 1e6  # Return in millions