import torch
from torch import nn
from torch import Tensor


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """内積注意の計算を行う.

        Args:
            query (Tensor): クエリ。shapeは(batch_size, query_len, d_model)。
            key (Tensor): キー。shapeは(batch_size, key_len, d_model)。
            value (Tensor): バリュー。shapeは(batch_size, value_len, d_model)。
        """
        # 1. query と key から、(batch_size, query_len, key_len)のスコアを計算
        score = torch.bmm(query, key.transpose(1, 2))

        # 2. 重みの和が1になるようにsoftmaxを計算
        weight = torch.softmax(score, dim=-1)

        # 3. value の重み付き和を計算
        output = torch.bmm(weight, value)

        return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor| None = None) -> Tensor:
        """スケール内積注意の計算を行う.

        Args:
            query (Tensor): クエリ。shapeは(batch_size, query_len, d_model)。
            key (Tensor): キー。shapeは(batch_size, key_len, d_model)。
            value (Tensor): バリュー。shapeは(batch_size, value_len, d_model)。
            mask (Tensor): Boolean マスク。shapeは(batch_size, query_len, key_len)。
        """
        d_k = query.size(-1)
        # query の次元 (= キーの次元) でスケーリング
        score = torch.bmm(query, key.transpose(1, 2)) / (d_k ** 0.5)

        # マスクがある場合は、-infを代入してsoftmaxの値が0になるようにする
        if mask is not None:
            score = score.masked_fill(mask, float("-inf"))


        weight = torch.softmax(score, dim=-1)
        output = torch.bmm(weight, value)

        return output
        

class AttentionHead(nn.Module):
    def __init__(self, d_k: int, d_v: int, d_model: int): 
        """MultiHeadAttentionのヘッド.

        Args:
            d_k (int): クエリ、キーの次元数
            d_v (int): バリューの次元数
            d_model (int): モデルの埋め込み次元数
        """
        super().__init__()
        # クエリ、キー、バリューを部分空間に埋め込むための全結合層
        self.linear_q = nn.Linear(d_model, d_k)
        self.linear_k = nn.Linear(d_model, d_k)
        self.linear_v = nn.Linear(d_model, d_v)

        self.attention = ScaledDotProductAttention()

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor| None = None) -> Tensor:
        """単一ヘッドのアテンションを計算する.

        Args:
            query (Tensor): クエリ。shapeは(batch_size, query_len, d_model)。
            key (Tensor): キー。shapeは(batch_size, key_len, d_model)。
            value (Tensor): バリュー。shapeは(batch_size, value_len, d_model)。

        Returns:
            Tensor: 出力。shapeは(batch_size, query_len, d_v)。
        """
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        output = self.attention(query, key, value, mask=mask)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_k: int, d_v: int, d_model: int): 
        """マルチヘッドアテンション.

        Args:
            n_heads (int): ヘッド数
            d_k (int): クエリ、キーの次元数
            d_v (int): バリューの次元数
            d_model (int): モデルの埋め込み次元数
        """
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_k, d_v, d_model) for _ in range(n_heads)])
        # 出力を変換する全結合層
        self.linear_o = nn.Linear(n_heads * d_v, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor|None = None) -> Tensor:
        """マルチヘッドアテンションを計算する.

        Args:
            query (Tensor): クエリ。shapeは(batch_size, query_len, d_model)。
            key (Tensor): キー。shapeは(batch_size, key_len, d_model)。
            value (Tensor): バリュー。shapeは(batch_size, value_len, d_model)。

        Returns:
            Tensor: 出力。shapeは(batch_size, query_len, d_model)。
        """
        # ヘッドごとにアテンションを計算
        head_out = [head(query, key, value, mask=mask) for head in self.heads]
        # ヘッドを結合
        head_out = torch.cat(head_out, dim=-1)
        # 出力を変換
        output = self.linear_o(head_out)
        return output

if __name__ == "__main__":
    dim = 16
    n_heads = 4
    batch_size = 2
    query_len, key_len, value_len = 3, 4, 4
    attention = ScaledDotProductAttention()
    multihead_attention = MultiHeadAttention(n_heads, dim, dim, dim)

    # 1. query と key から、(batch_size, query_len, key_len)のスコアを計算
    query = torch.randn(batch_size, query_len, dim)
    key = torch.randn(batch_size, key_len, dim)
    value = torch.randn(batch_size, value_len, dim)

    output = attention(query, key, value)
    output2 = multihead_attention(query, key, value)
    print(output.shape)  # torch.Size([batch_size, query_len, dim])
    print(output2.shape)  # torch.Size([batch_size, query_len, dim])