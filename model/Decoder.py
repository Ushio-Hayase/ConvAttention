import torch
import torch.nn as nn

from MultiHeadAttention import MultiHeadAttentionLayer
from FFNN import FFNN


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, dff: int, num_heads: int, dropout: float):
        super(DecoderBlock, self).__init__()

        self.attention1 = MultiHeadAttentionLayer(d_model, num_heads)
        self.attention2 = MultiHeadAttentionLayer(d_model, num_heads)
        self.ffnn = FFNN(d_model, dff)
        
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm()

    def forward(self, inputs: torch.Tensor, encoder_inputs: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Decoder Block 

        mask must be Boolean Tensor
        """
        inputs1 = self.norm(inputs)
        inputs1 = self.attention1(inputs1, inputs1, inputs1, mask)
        inputs1 = self.drop(inputs1) + inputs

        inputs2 = self.norm(inputs1)
        inputs2 = self.attention2(inputs2, encoder_inputs, encoder_inputs)
        inputs2 = self.drop(inputs2) + inputs1

        inputs3 = self.norm(inputs2)
        inputs3 = self.ffnn(inputs3)
        inputs3 = self.drop(inputs3) + inputs2

        return inputs3


class Decoder(nn.Module):
    def __init__(self, d_model: int, dff: int, num_heads: int, num_layers: int, dropout: float):
        super(Decoder, self).__init__()

        self.decoder = nn.ModuleList([DecoderBlock(d_model, dff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, inputs: torch.Tensor, encoder_inputs: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Mask argument must be Boolean Tensor
        """
        for decoder_block in self.decoder:
            inputs = decoder_block(inputs, mask) # 디코더 레이어 개수만큼 반복

        return inputs
        

    

