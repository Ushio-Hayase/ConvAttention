import torch
import torch.nn as nn

from FFNN import FFNN
from MultiHeadAttention import MultiHeadAttentionLayer


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, dff: int, num_heads: int, dropout: float):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttentionLayer(d_model, num_heads)
        self.ffnn = FFNN(d_model, dff)
        
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm()

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Encoder Block
        
        mask must be Boolean Tensor
        """
        inputs1 = self.norm(inputs)
        inputs1 = self.attention(inputs1, inputs1, inputs1, mask)
        inputs1 = self.drop(inputs1) + inputs

        inputs2 = self.norm(inputs1)
        inputs2 = self.ffnn(inputs2)
        inputs2 = self.drop(inputs2) + inputs1

        return inputs2
    
class Encoder(nn.Module):
    def __init__(self, d_model: int, dff: int,
                num_heads: int, num_layers: int, dropout: int):
        super(Encoder, self).__init__()

        self.encoder = nn.ModuleList([EncoderBlock(d_model, dff, num_heads, dropout) 
                                      for _ in range(num_layers)])

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Mask argument must be Boolean Tensor
        """
        for layer in self.encoder:
            inputs = layer(inputs, mask)

        return inputs
