import math
import torch
import torch.nn as nn

from .PositionalEncoding import PositionalEncoding
from .Encoder import Encoder
from .Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, d_model: int, dff: int, num_heads: int, 
                num_layers: int, max_len: int, vocab_size: int, dropout: float, pad_idx: int, device):
        super(Transformer, self).__init__()

        self.pos_enc = PositionalEncoding(d_model, dropout,max_len)
        self.embd = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.enc = Encoder(d_model, dff, max_len, num_heads, num_layers, dropout)
        self.dec = Decoder(d_model, dff, max_len, num_heads, num_layers, dropout)
        self.fc = nn.Linear(d_model, vocab_size)

        self.device = device
        self.pad_idx = pad_idx
        self.d_model = d_model


    def forward(self, enc_inputs: torch.Tensor, dec_inputs: torch.Tensor) -> torch.Tensor:
        enc_mask = self.create_padding_mask(enc_inputs)
        dec_mask = self.create_attention_mask(dec_inputs) | self.create_padding_mask(dec_inputs)


        enc_inputs = self.embd(enc_inputs) * math.sqrt(self.d_model)
        dec_inputs = self.embd(dec_inputs) * math.sqrt(self.d_model)

        enc_inputs = self.pos_enc(enc_inputs)
        dec_inputs = self.pos_enc(dec_inputs)

        enc_out = self.enc(enc_inputs, enc_mask)
        dec_out = self.dec(dec_inputs, enc_out, dec_mask, enc_mask)

        return self.fc(dec_out)



    def create_padding_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs == self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_attention_mask(self, trg: torch.Tensor) -> torch.Tensor:
        trg_len = trg.size(1)
        return torch.triu(torch.ones(trg_len, trg_len), diagonal=1).bool().to(self.device)
        
