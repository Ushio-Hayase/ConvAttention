import torch
import torch.nn as nn

from PositionalEncoding import PositionalEncoding
from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, d_model: int, dff: int, num_heads: int, 
                num_layers: int, max_len: int, vocab_size: int, dropout: float, pad_idx: int, device):
        super(Transformer, self).__init__()

        self.pos_enc = PositionalEncoding(d_model, max_len, device=device)
        self.embd = nn.Embedding(vocab_size, d_model, pad_idx)

        self.enc = Encoder(d_model, dff, num_heads, num_layers, dropout)
        self.dec = Decoder(d_model, dff, num_heads, num_layers, dropout)
        self.fc = nn.Linear(d_model, vocab_size)

        self.device = device
        self.pad_idx = pad_idx


    def forward(self, enc_inputs: torch.Tensor, dec_inputs: torch.Tensor) -> torch.Tensor:
        enc_mask = self.create_padding_mask(enc_inputs)
        dec_mask = self.create_attention_mask(dec_inputs.size(1)) + self.create_padding_mask(dec_inputs)

        enc_inputs = self.embd(enc_inputs)
        dec_inputs = self.embd(dec_inputs)

        enc_inputs = self.pos_enc(enc_inputs)
        dec_inputs = self.pos_enc(dec_inputs)

        enc_out = self.enc(enc_inputs, enc_mask)
        dec_out = self.dec(dec_inputs, enc_out, dec_mask)

        return self.fc(dec_out)



    def create_padding_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs == self.pad_idx)
    
    def create_attention_mask(self, size: int) -> torch.Tensor:
        return torch.triu(torch.ones((size, size)), diagonal=1)
