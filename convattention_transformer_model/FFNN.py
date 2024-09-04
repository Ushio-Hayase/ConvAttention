import torch
import torch.nn as nn


class FFNN(nn.Module):
    def __init__(self, d_model: int, dff: int):
        super(FFNN, self).__init__()

        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.fc1(inputs)
        inputs = nn.functional.gelu(inputs)
        inputs = self.fc2(inputs)

        return inputs
    