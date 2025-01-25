import torch
import torch.nn as nn
import torch.functional as F

class DoubleConvolution(nn.Module):
    """
    PyTorch Module that performs a double convolution on the input. 
    This module consists of a conovolution followed by a rectified 
    linear unit (ReLU) two times.
    """

    def __init__(self, in_channels, out_channels):
        super.__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)
