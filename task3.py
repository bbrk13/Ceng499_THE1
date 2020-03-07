import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
  num_channels = 1
  num_outputs = 10

  def __init__(self):
    super().__init__()
    # write your code here

  def forward(self, x):
    # write your code here
    return NotImplementedError
