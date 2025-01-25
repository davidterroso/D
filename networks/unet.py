import torch
import torch.nn as nn
import torch.functional as F

class DoubleConvolution(nn.Module):
    """
    PyTorch Module that performs a double convolution on the input. 
    This module consists of a convolution followed by a rectified 
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
        # Calls nn.Module class
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
            x (PyTorch tensor): network input or result 
            of the downsample operation before

        Return:
            (PyTorch tensor): result of the double 
            convolution operation on the input x
        """
        return self.conv_op(x)

class DownSample(nn.Module):
    """
    PyTorch Module that downsamples the input. 
    This module consists of a double convolution
    followed by a max pooling operation.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initiates the DownSample object, which is composed of a double convolution 
        and a Max Pooling operation
        
        Args:
            in_channels (int): Number of channels that are input in this module
            out_channels (int): Number of channels that are output of this module
        
        Return:
            None
        """
        # Calls nn.Module class
        super().__init__()
        # Calls the DoubleConvolution function
        self.conv = DoubleConvolution(in_channels, out_channels)
        # Calls the MaxPooling function
        # Shape (input): ((h - 2 - 2) x (w - 2 - 2) x out_channels)
        # Shape (output): ((h - 2 - 2) / 2 x (w - 2 - 2) / 2 x out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward step of the down sample, returning
        its result when applied to an input x

        Args: 
            self (DownSample object): the 
            DownSample object which is going
            to be performed on input x
            x (PyTorch tensor): result of the
            double convolution

        Return:
            (PyTorch tensor): result of the down 
            sampling operation on the input x
            (PyTorch tensor): result of the max 
            pooling operation on the input x
        """
        # Calculates the double convolution applied 
        # to the x input
        down = self.conv(x)
        # Calculates the max pooling operation 
        # applied to the result of the double 
        # convolution
        pool = self.pool(down)
        # Returns the result of the downsampling 
        # and max pooling operation
        return down, pool
    