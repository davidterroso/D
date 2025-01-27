from torch import cat
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, ReLU, MaxPool2d

class DoubleConvolution(Module):
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
        super().__init__()
        # Defines the double convolution
        self.conv_op = Sequential(
            # Two dimensional convolution with kernel of size three 
            # and padding one. 
            # Shape (input): (h x w x in_channels)
            # Shape (output): ((h - 2) x (w - 2) x out_channels)
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Rectified Linear Activation function. Does not change the 
            # input shape. 
            # inplace argument declares a change directly on the input, 
            # without producing an output
            ReLU(inplace=True),
            # Shape (input): ((h - 2) x (w - 2) x out_channels)
            # Shape (output): ((h - 2 - 2) x (w - 2 - 2) x out_channels)
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward step of the double convolution, 
        returning its result when applied to an input x

        Args: 
            self (DoubleConvolution object): the 
            DoubleConvolution object which is going
            to be performed on input x
            x (PyTorch tensor): result operation
            before, which can be a downsample of a
            upsample operation

        Return:
            (PyTorch tensor): result of the double 
            convolution operation on the input x
        """
        return self.conv_op(x)

class DownSample(Module):
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
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward step of the downsample, returning
        its result when applied to an input x

        Args: 
            self (DownSample object): the 
            DownSample object which is going
            to be performed on input x
            x (PyTorch tensor): result of the
            double convolution

        Return:
            (PyTorch tensor): result of the 
            downsampling operation on the input x
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

class UpSample(Module):
    """
    PyTorch Module that upsamples the input. 
    This module consists of a de-convolution
    followed by a concatenation with a cropped
    output of a previous double convolution.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initiates the UpSample object, which is composed of a de-convolution 
        and a concatenation with a cropped output of a previous double convolution.
        
        Args:
            in_channels (int): Number of channels that are input in this module
            out_channels (int): Number of channels that are output of this module
        
        Return:
            None
        """
        # Calls nn.Module class
        super().__init__()
        # Initiates the de-convolution function
        # Shape (input): (h x w x in_channels)
        # Shape (output): (h x w x in_channels // 2)
        self.up = ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Initiates the double convolution function
        # Shape (input): (2*h x 2*w x in_channels // 2)
        # Shape (output): ((2*h - 2 - 2) x (2*w - 2 - 2) x in_channels // 2)
        self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward step of the upsample, returning
        its result when applied to an input x1,
        that results from the downsampling step, 
        and x2

        Args: 
            self (DownSample object): the 
            DownSample object which is going
            to be performed on input x
            x1 (PyTorch tensor): result of the
            previous step
            x2 (PyTorch tensor): result of the
            downsampling step

        Return:
            (PyTorch tensor): result of the upsample, 
            concatenation, and convolution
        """
        # Calculates the up convolution applied 
        # to the x1 input
        x1 = self.up(x1)
        # Concatenates the results of the
        # upconvolution with those obtained in 
        # the downsampling step 
        x = cat([x1, x2], 1)
        # Returns the result of the convolution
        # applied to the result of the concatenation 
        # step
        return self.conv(x)
    
class UNet(Module):
    """
    PyTorch Module that pieces together the 
    modules created above, forming a U-Net. 
    """
    def __init__(self, in_channels, num_classes):
        """
        Initiates the UNet object, indicating the input and output size of the
        multiple models and the order in which they are presented
        
        Args:
            in_channels (int): Number of channels that are input in this module
            out_channels (int): Number of channels that are output of this module
        
        Return:
            None
        """
        # Calls the nn.Module class
        super().__init__()

        # Convolutions that form the encoding path
        self.down_convolution_1 = DownSample(in_channels, 64) # (h, w, 1) -> ((h - 4) / 2, (w - 4) / 2, 64)
        self.down_convolution_2 = DownSample(64, 128) # ((h - 4) / 2, (w - 4) / 2, 64) -> (((h - 4) / 2 - 4) / 2, ((w - 4) / 2 - 4) / 2, 128)
        self.down_convolution_3 = DownSample(128, 256) # (((h - 4) / 2 - 4) / 2, ((w - 4) / 2 - 4) / 2, 128) -> ((((h - 4) / 2 - 4) / 2 - 4) / 2, (((w - 4) / 2 - 4) / 2 - 4) / 2, 256)
        self.down_convolution_4 = DownSample(256, 512) # ((((h - 4) / 2 - 4) / 2 - 4) / 2, (((w - 4) / 2 - 4) / 2 - 4) / 2, 256) -> (((((h - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, ((((w - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, 512)

        # Double convolution on the bottleneck
        bottle_neck = self.bottle_neck = DoubleConvolution(512, 1024) # (((((h - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, ((((w - 4) / 2 - 4 / 2) - 4) / 2 - 4) / 2, 512) -> ((((((h - 4) / 2 - 4 / 2 - 4 / 2 - 4) / 2 - 4) / 2, ((((((w - 4) / 2 - 4) / 2 - 4) / 2) - 4) / 2 - 4) / 2, 1024)

        # Up-convolutions that form the decoding path
        self.up_convolution_1 = UpSample(1024, 512) # ((((((h - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, ((((((w - 4) / 2 - 4) / 2 - 4) / 2) - 4) / 2 - 4) / 2, 1024) -> (((((h - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, ((((w - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, 512)
        self.up_convolution_2 = UpSample(512, 256) # (((((h - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, ((((w - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, 512) -> ((((h - 4) / 2 - 4) / 2 - 4) / 2, (((w - 4) / 2 - 4) / 2 - 4) / 2, 256)
        self.up_convolution_3 = UpSample(256, 128) # ((((h - 4) / 2 - 4) / 2 - 4) / 2, (((w - 4) / 2 - 4) / 2 - 4) / 2, 256) -> (((h - 4) / 2 - 4) / 2, ((w - 4) / 2 - 4) / 2, 128)
        self.up_convolution_4 = UpSample(128, 64) # (((h - 4) / 2 - 4) / 2, ((w - 4) / 2 - 4) / 2, 128) -> ((h - 4) / 2, (w - 4) / 2, 64)

        # Last convolution to obtain the segmentation masks
        self.out = Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1) # ((h - 4) / 2, (w - 4) / 2, 64) -> (h, w, num_classes)

    def forward(self, x):
        """
        Forward step of the U-Net, returning
        the segmented masks.

        Args: 
            self (UNet object): the 
            UNet object that contains the layers
            x (PyTorch tensor): input image

        Return:
            (PyTorch tensor): image with the masks 
            as other channels
        """
        # Performs the steps in the encoding path of the 
        # network
        down_1, pool_1 = self.down_convolution_1(x)
        down_2, pool_2 = self.down_convolution_2(pool_1)
        down_3, pool_3 = self.down_convolution_3(pool_2)
        down_4, pool_4 = self.down_convolution_4(pool_3)

        # Calculates the last convolution before 
        # upsampling without max pooling
        b = self.bottle_neck(pool_4)

        # Performs the steps in the decoding path 
        # of the network
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        # Performs the final convolution that outputs
        # the image with its respective masks in 
        # different channels, which is then returned
        out = self.out(up_4)

        return out
