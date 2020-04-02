import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module

def standardize(x, eps=1e-6):

    x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + eps)

    return x_norm
    

class Norm(Module):
    def __init__(self, input_size, eps=1e-6):
        super().__init__()
        self.size = input_size        
        # create two learnable parameters to calibrate normalisation
        self.std = nn.Parameter(torch.ones(self.size))
        self.mean = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.std * standardize(x, self.eps) + self.mean
        return norm


if __name__ == '__main__':

    input_size = 1000
    x = torch.rand(100, input_size)

    norm = Norm(input_size)

    x_norm = norm(x)

    print(x_norm.mean(-1).data.detach().numpy())
    print(x_norm.std(-1).data.detach().numpy())
