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
        """
        Initiates the DoubleConvolution object, which is composed of two 
        sets of one two-dimensional convolution followed by a ReLU function
        
        Args:
            in_channels (int): Number of channels that are input in this module
            out_channels (int): Number of channels that are output of this module
        
        Return:
            None
        """
        super.__init__()
        # Defines the double convolution
        self.conv_op = nn.Sequential(
            # Two dimensional convolution with kernel of size three 
            # and padding one. 
            # Shape (input): (h x w x in_channels)
            # Shape (output): ((h - 2) x (w - 2) x out_channels)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Rectified Linear Activation function. Does not change the 
            # input shape. 
            # inplace argument declares a change directly on the input, 
            # without producing an output
            nn.ReLU(inplace=True),
            # Shape (input): ((h - 2) x (w - 2) x out_channels)
            # Shape (output): ((h - 2 - 2) x (w - 2 - 2) x out_channels)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward step of the double convolution, 
        returning its result when applied to an input x

        Args: 
            self (DoubleConvolution object): the 
            DoubleConvolution object which is going
            to be performed on input x
            x (PyTorch tensor): network input

        Return:
            (PyTorch tensor): result of the double 
            convolution operation on the input x
        """
        return self.conv_op(x)
