import torch
import torch.nn as nn

from qkv import QKVAttention

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k: int = d_model // num_heads

        self.scaled_dot_product_attention = QKVAttention()

        self.w_q: nn.Linear = nn.Linear(d_model, self.d_k)
        self.w_k: nn.Linear = nn.Linear(d_model, self.d_k)
        self.w_v: nn.Linear = nn.Linear(d_model, self.d_k)

        self.out_fc: nn.Linear = nn.Linear(d_model, d_model)

    def _split_heads(self, inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
        inputs = inputs.view([batch_size, -1, self.num_heads, self.d_k]) # batch_size, seq_len, d_model -> batch_size, seq_len, num_headsd, d_k
        return inputs.permute(0, 2, 1, 3) # batch_size, num_heads, seq_len, d_k
        
    def concat(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs Tensor shape : batch_size, num_heads, seq_len, d_k

        return Tensor shape : batch_size, seq_len, d_model or seq_len, batch_size, d_model
        """
        batch_size, num_heads, seq_len, d_k= inputs.size()

        return inputs.transpose(1, 2).contiguous().view(batch_size, seq_len, d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        mask에는 마스킹할 자리가 True인 Bool 형태의 텐서가 들어와야 합니다.
        """
        batch_size: int = None

        # 배치 사이즈 구하기
        if self.batch_first:
            batch_size = query.size(0)
        else:
            batch_size = query.size(1)

        # 차원 변환
        query = self.w_q(query) 
        key = self.w_k(key)
        value = self.w_v(value)

        # num heads에 따라 텐서 분리
        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)

        out, attention_score = self.scaled_dot_product_attention(query, key, value, mask) # QKV 어텐션

        out = self.concat(out) # 분할했던 텐서 병합

        return self.out_fc(out) # 출력 full-connectly layer