import torch
import torch.utils
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y) -> None:
        self.x = data_x
        self.y = data_y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
