import math
import torch
import torch.nn as nn


class QKVAttention(nn.Module):
    """
    execute scaled dot product attention

    q : query
    k : key
    v : value
    """
    def __init__(self):
        super(QKVAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None):
        """
        mask 인자에는 Bool 형태의 텐서가 넣어져야 합니다
        """

        batch_size: int = q.size(0) # 배치 사이즈
        d_k: int = q.size(3)


        k_t = k.transpose(2, 3) # batch_size, num_heads, d_k, seq_len
        attention_score = torch.matmul(q, k_t) / math.sqrt(d_k)

        if mask is not None:
            attention_score.masked_fill_(mask, -1e6) # 마스킹

        attention_score: torch.Tensor = self.softmax(attention_score)

        v = attention_score * v

        return v, attention_score
