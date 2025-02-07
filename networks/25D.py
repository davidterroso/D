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
        self.conv_op = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=7, padding="same"),
            ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv_op(x)
        