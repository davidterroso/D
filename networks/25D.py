from torch import cat
from torch.nn.functional import softmax
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, ReLU, MaxPool2d, Dropout2d, BatchNorm2d, Upsample

class InitialConvolution(Module):
    def __init__(self, in_channels, out_channels):
        """
        Initiates the InitialConvolution object, which is composed of one 
        two-dimensional convolution with kernel size seven followed by a 
        ReLU function
        
        Args:
            in_channels (int): Number of channels that are input in this module
            out_channels (int): Number of channels that are output of this module
        
        Return:
            None
        """
        # Calls nn.Module class
        super().__init__()
        # Defines the initial convolution
        self.conv_op = Sequential(
            # Two dimensional convolution with kernel of size seven 
            # and padding same. 
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
            # Shape (input): (h x w x in_channels)
            # Shape (output): ((h - 2) x (w - 2) x out_channels)
            Conv2d(in_channels, out_channels, kernel_size=3),
            # Rectified Linear Activation function. Does not change the 
            # input shape. 
            # inplace argument declares a change directly on the input, 
            # without producing an output
            ReLU(inplace=True),
            # Shape (input): ((h - 2) x (w - 2) x out_channels)
            # Shape (output): ((h - 2 - 2) x (w - 2 - 2) x out_channels)
            Conv2d(out_channels, out_channels, kernel_size=3),
            # Applies batch normalization to the channels, 
            # not changing the output shapes
            BatchNorm2d(out_channels),
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
    def __init__(self, in_channels, out_channels, p):
        """
        Initiates the UpSample object, which is composed of a de-convolution 
        and a concatenation with a cropped output of a previous double convolution.
        
        Args:
            in_channels (int): Number of channels that are input in this module
            out_channels (int): Number of channels that are output of this module
            p (float): Probability of dropout 
        
        Return:
            None
        """
        # Calls nn.Module class
        super().__init__()        
        # Initiates the de-convolution function
        # Shape (input): (h x w x in_channels)
        # Shape (output): (h x w x in_channels // 2)
        self.up = Sequential(
            ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, padding="same"),
            Dropout2d(p=p)
        )
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
    
class TennakoonUNet(Module):
    """
    PyTorch Module that pieces together the 
    modules created above, forming a PyTorch 
    module of the network implemented by 
    Tennakoon et al., 2018. 
    """
    def __init__(self, in_channels, num_classes):
        """
        Initiates the TennakoonUNet object, indicating the input and output size of the
        multiple models and the order in which they are presented
        
        Args:
            in_channels (int): Number of channels that are input in this module
            num_classes (int): Number of classes that are expected to output 
                from this network
        
        Return:
            None
        """
        # Calls the nn.Module class
        super().__init__()

        self.n_channels = in_channels
        self.n_classes = num_classes

        # Initial convolution
        self.initial_convolution = InitialConvolution(in_channels, 64)

        # Convolutions that form the encoding path
        self.down_convolution_1 = DownSample(64, 64) # (h, w, 1) -> ((h - 4) / 2, (w - 4) / 2, 64)
        self.down_convolution_2 = DownSample(64, 128) # ((h - 4) / 2, (w - 4) / 2, 64) -> (((h - 4) / 2 - 4) / 2, ((w - 4) / 2 - 4) / 2, 128)
        self.down_convolution_3 = DownSample(128, 256) # (((h - 4) / 2 - 4) / 2, ((w - 4) / 2 - 4) / 2, 128) -> ((((h - 4) / 2 - 4) / 2 - 4) / 2, (((w - 4) / 2 - 4) / 2 - 4) / 2, 256)

        # Double convolution on the bottleneck
        self.bottle_neck = DoubleConvolution(256, 512) # # ((((h - 4) / 2 - 4) / 2 - 4) / 2, (((w - 4) / 2 - 4) / 2 - 4) / 2, 256) -> (((((h - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, ((((w - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, 512)

        # Up-convolutions that form the decoding path
        self.up_convolution_1 = UpSample(512, 256, p=0.25) # (((((h - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, ((((w - 4) / 2 - 4) / 2 - 4) / 2 - 4) / 2, 512) -> ((((h - 4) / 2 - 4) / 2 - 4) / 2, (((w - 4) / 2 - 4) / 2 - 4) / 2, 256)
        self.up_convolution_2 = UpSample(256, 128, p=0.25) # ((((h - 4) / 2 - 4) / 2 - 4) / 2, (((w - 4) / 2 - 4) / 2 - 4) / 2, 256) -> (((h - 4) / 2 - 4) / 2, ((w - 4) / 2 - 4) / 2, 128)
        self.up_convolution_3 = UpSample(128, 64, p=0.5) # (((h - 4) / 2 - 4) / 2, ((w - 4) / 2 - 4) / 2, 128) -> ((h - 4) / 2, (w - 4) / 2, 64)

        # Dropout method for the cropped outputs, 
        # depending on the probabilities of dropout
        self.cropped_dropout_25 = Dropout2d(0.25)
        self.cropped_dropout_50 = Dropout2d(0.50)

        # Used to include information from larger feature maps, through Upsampling
        # Do not confuse the functions Upsample with UpSample
        self.multiscale_up_1 = Upsample(8, 8)
        self.multiscale_up_2 = Upsample(4, 4)
        self.multiscale_up_3 = Upsample(2, 2)

        # Last convolution to obtain the segmentation masks
        self.out = Sequential(
            Conv2d(in_channels=4, out_channels=num_classes, kernel_size=3, padding="same"), # ((h - 4) / 2, (w - 4) / 2, 64) -> (h, w, num_classes)
            ReLU(inplace=True),
            softmax(dim=1)
        )

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
        # network and applies both cropping and dropout to
        # the results of the downsampling
        down_0 = self.initial_convolution(x)
        down_1, pool_1 = self.down_convolution_1(down_0)
        down_1_cropped = down_1[:, :, 5:-4, 5:-4]
        down_1_cropped = self.cropped_dropout_50(down_1_cropped)
        down_2, pool_2 = self.down_convolution_2(pool_1)
        down_2_cropped = down_2[:, :, 17:-17, 17:-17]
        down_2_cropped = self.cropped_dropout_25(down_2_cropped)
        down_3, pool_3 = self.down_convolution_3(pool_2)
        down_3_cropped = down_3[:, :, 42:-42, 42:-42]
        down_3_cropped = self.cropped_dropout_25(down_3_cropped)

        # Calculates the last convolution before 
        # upsampling, without max pooling
        b = self.bottle_neck(pool_3)

        # Performs the steps in the decoding path 
        # of the network
        up_1 = self.up_convolution_1(b, down_3_cropped)
        up_2 = self.up_convolution_2(up_1, down_2_cropped)
        up_3 = self.up_convolution_3(up_2, down_1_cropped)

        # Calculates the upsampling operations
        ms_1 = self.multiscale_up_1(down_3)
        ms_1 = ms_1[:, :, 14:-14, 14:-14]
        ms_2 = self.multiscale_up_2(up_1)
        ms_2 = ms_2[:, :, 6:-6, 6:-6]
        ms_3 = self.multiscale_up_3(up_2)
        ms_3 = ms_3[:, :, 2:-2, 2:-2]

        # Merges the results from the different upsamplings
        merge = cat([ms_1, ms_2, ms_3, up_3], dim=1)

        # Performs the final convolution that outputs
        # the image with its respective masks in 
        # different channels, which is then returned
        out = self.out(merge)

        return out
        