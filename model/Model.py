import torch

from Patch_Embedding import Patch_Embd
from Decoder import Decoder


class DTrOCR(torch.nn.Module):
    def __init__(self, d_model: int, dff: int, num_heads: int, num_layers: int, 
                 patch_size: int, channels: int, image_size: int, dropout: float, 
                 batch_first: bool):
        
        self.patch_embd = Patch_Embd(patch_size, channels, d_model, image_size, batch_first)
        self.Decoder = Decoder(d_model, dff, num_heads, num_layers, dropout, batch_first)
        self.batch_first = batch_first

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        preprocessed_data: torch.Tensor = self.patch_embd(inputs)
        if self.batch_first:
            return self.Decoder(preprocessed_data)
        else:
            preprocessed_data = preprocessed_data.permute(1, 0, 2)
            return self.Decoder(preprocessed_data)
