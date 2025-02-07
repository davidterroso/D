from torch import cat
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, ReLU, MaxPool2d, Dropout2d, BatchNorm2d

class InitialConvolution(Module):
    def __init__(self, in_channels, out_channels):
        """
        Initiates the InitialConvolution object, which is composed of one 
        two-dimensional convolution followed by a ReLU function
        
        Args:
            in_channels (int): Number of channels that are input in this module
            out_channels (int): Number of channels that are output of this module
        
        Return:
            None
        """
        # Calls nn.Module class
        super().__init__()
        # Defines the double convolution
        self.conv_op = Sequential(
            # Two dimensional convolution with kernel of size seven 
            # and padding one. 
            # Shape (input): (h x w x in_channels)
            # Shape (output): (h x w x out_channels), because padding is "same"
            Conv2d(in_channels, out_channels, kernel_size=7, padding="same"),
            # Rectified Linear Activation function. Does not change the 
            # input shape. 
            # inplace argument declares a change directly on the input, 
            # without producing an output
            ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward step of the first convolution, 
        returning its result when applied to an input x

        Args: 
            self (InitialConvolution object): the 
                InitialConvolution object which is going
                to be performed on input x
            x (PyTorch tensor): result operation
                before, which can be a downsample of a
                upsample operation

        Return:
            (PyTorch tensor): result of the first 
                convolution operation on the input x
        """
        return self.conv_op(x)
        