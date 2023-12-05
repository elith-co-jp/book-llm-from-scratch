import torch
from torch import nn
from torch import Tensor


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        """内積注意の計算を行う.

        Args:
            query (Tensor): クエリ。shapeは(batch_size, query_len, dim)。
            key (Tensor): キー。shapeは(batch_size, key_len, dim)。
            value (Tensor): バリュー。shapeは(batch_size, value_len, dim)。
        """
        # 1. query と key から、(batch_size, query_len, key_len)のスコアを計算
        score = torch.bmm(query, key.transpose(1, 2))

        # 2. 重みの和が1になるようにsoftmaxを計算
        weight = torch.softmax(score, dim=-1)

        # 3. value の重み付き和を計算
        output = torch.bmm(weight, value)

        return output
        

if __name__ == "__main__":
    attention = DotProductAttention()

    # 1. query と key から、(batch_size, query_len, key_len)のスコアを計算
    query = torch.randn(2, 3, 5)
    key = torch.randn(2, 4, 5)
    value = torch.randn(2, 4, 5)

    output = attention(query, key, value)
    print(output.shape)  # torch.Size([2, 3, 5])