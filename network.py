import torch
from torch import nn 
import numpy as np


class InitNet(nn.Module):

    def __init__(self, m, n):
        super(InitNet, self).__init__()
        input_size = int(2 * m * (1+n))
        output_size = int(2*n)
        middle_1 = int(np.floor(input_size/2))
        middle_2 = int(np.floor(middle_1/2))
        if middle_2 < output_size:
            middle_2 = output_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, middle_1),
            nn.Tanh(),
            nn.Linear(middle_1, middle_2),
            nn.Tanh(),
            nn.Linear(middle_2, output_size),
        )

    def forward(self, y):
        xhat = self.layers(y)
        return xhat