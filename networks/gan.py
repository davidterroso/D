from torch import cat
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, Dropout2d, Module, LeakyReLU, ReLU, Sequential, Sigmoid, Tanh

class Generator(Module):
    """
    PyTorch Module that generates the 
    intermediate image from the input 
    set of images.
    """
    def __init__(self, number_of_channels: int=1, 
                 feature_map_size: int=64):
        """
        Initiates the Generator object, which 
        encodes the image into its latent state
        to which noise is added before being 
        decoded.
        
        Args:
            self (Generator object): the object 
                Generator that will now be 
                defined
            number_of_classes (int): number of 
                output channels. In the 
                discriminator, we aim to 
                generate a grayscale image 
                from two other grayscale images
            feature_map_size (int): number of 
                channels after the first 
                convolution. The default value 
                is 64, which was the value being 
                used in the adapted code  
        
        Return:
            None
        """
        # Starts the initialization of the 
        # Generator object
        super(Generator, self).__init__()

        # Defines the down-sampling blocks
        self.down_sample_blocks = Sequential(
            # N = number_of_classes
            # F = feature_map_size
            # Input Shape: B x (N * 2) x H x W
            Conv2d(number_of_channels * 2, feature_map_size * 2, kernel_size=3, stride=1, padding=1, bias=False),
            # Shape: B x (F * 2) x H x W
            BatchNorm2d(feature_map_size * 2), # Same shape
            LeakyReLU(0.02, inplace=True), # Same shape
            Conv2d(feature_map_size * 2, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            # Shape: B x (F * 2) x (H / 2) x (W / 2)
            BatchNorm2d(feature_map_size * 2), # Same shape
            LeakyReLU(0.02, inplace=True), # Same shape
            Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            # Shape: B x (F * 4) x (H / 4) x (W / 4)
            BatchNorm2d(feature_map_size * 4), # Same shape
            LeakyReLU(0.02, inplace=True), # Same shape
            Conv2d(feature_map_size * 4, feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            # Shape: B x (F * 8) x (H / 8) x (W / 8)
            BatchNorm2d(feature_map_size * 8), # Same shape
            LeakyReLU(0.02, inplace=True) # Same shape
            # Final Shape: B x (F * 8) x (H / 8) x (W / 8)
        )
        
        # Defines the up-sampling blocks
        self.up_sample_block = Sequential(
            # Input Shape: B x (F * 8) x (H / 8) x (W / 8)
            ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            # Shape: B x (F * 4) x (H / 4) x (W / 4)
            BatchNorm2d(feature_map_size * 4), # Same shape
            LeakyReLU(0.02, inplace=True), # Same shape
            ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            # Shape: B x (F * 2) x (H / 2) x (W / 2)
            BatchNorm2d(feature_map_size * 2), # Same shape
            LeakyReLU(0.02, inplace=True), # Same shape
            ConvTranspose2d(feature_map_size * 2, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            # Shape: B x F x H x W
            BatchNorm2d(feature_map_size), # Same shape
            LeakyReLU(0.02, inplace=True), # Same shape
            ConvTranspose2d(feature_map_size, number_of_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # Shape: B x N x H x W
            Tanh() # Same shape
            # Final Shape: B x N x H x W
        )

    def forward(self, img_before, img_after):
        """
        Forward step of the generator, 
        returning its result when 
        applied to input image

        Args: 
            self (Generator object): the 
                Generator object
            img_before (PyTorch tensor): 
                image before the one we 
                are looking to generate
            img_after (PyTorch tensor): 
                image before the one we 
                are looking to generate

        Return:
            (PyTorch tensor): result of the 
                operations applied in the 
                forward step to the input x
        """      
        # Stacks the previous image and the image after 
        # along the second dimension
        input_stack = cat((img_before, img_after), 1)
        # Performs the down-sampling on the input stack
        out_down = self.down_sample_blocks(input_stack)
        # Performs the up-sampling on the input stack
        out_up = self.up_sample_block(out_down)
        return out_up
    
class Discriminator(Module):
    """
    PyTorch Module that discriminates whether 
    the input set of three consecutive images,
    is real or fake.
    """
    def __init__(self, number_of_classes: int=1, 
                 feature_map_size: int=64):
        """
        Initiates the Discriminator object, which 
        is composed of multiple convolutions and 
        a linear transformation followed by a 
        sigmoid that results in a final binary 
        prediction.
        
        Args:
            self (Discriminator object): the 
                object Discriminator that will 
                now be defined
            number_of_classes (int): number of 
                input channels. In the 
                discriminator, we aim to 
                classify a grayscale image 
                either as fake or real so the 
                default value is one. This 
                value also corresponds to the 
                number of output channels of 
                the Generator 
            feature_map_size (int): number of 
                channels after the first 
                convolution. The default value 
                is 64, which was the value being 
                used in the adapted code  
        
        Return:
            None
        """
        # Starts the initialization of the 
        # Discriminator object
        super(Discriminator, self).__init__()
        
        # In this part, a series of convolutions, 
        # combined with dropout, batch normalization, 
        # and ReLU, are applied to the input image, 
        # resulting in a prediction of size one
        # Since this architecture is Fully 
        # Convolutional (FCN), it is agnostic to the 
        # input shape and an image of any shape can 
        # be used to train it
        self.conv_blocks = Sequential(
            # Input shape: C x H x W
            Conv2d(number_of_classes, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            LeakyReLU(0.2, inplace=True),
            Dropout2d(0.25),
            # Current shape: feature_map_size x (H / 2) x (W / 2)
            Conv2d(feature_map_size, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(feature_map_size * 2),
            LeakyReLU(0.2, inplace=True),
            Dropout2d(0.25),
            # Current shape: (feature_map_size * 2) x (H / 4) x (W / 4)
            Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(feature_map_size * 4),
            LeakyReLU(0.2, inplace=True),
            Dropout2d(0.25),
            # Current shape: (feature_map_size * 4) x (H / 8) x (W / 8)
            Conv2d(feature_map_size * 4, feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(feature_map_size * 8),
            LeakyReLU(0.2, inplace=True),
            Dropout2d(0.25),
            # Current shape: (feature_map_size * 8) x (H / 16) x (W / 16)
            Conv2d(feature_map_size * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
            # Final shape: 1 x (H / 16 - 3) x (W / 16 - 3)
            # For an input shape of: 1 x 496 x 512, it would result in an output shape of 
            # 1 x 28 x 29
        )

    def forward(self, img_mid):
        """
        Forward step of the discriminator, 
        that returns the logits that 
        correspond to the probability of the 
        image being real

        Args: 
            self (Generator object): the 
                Generator object
            img_mid (PyTorch tensor): 
                image we are looking to
                classify

        Return:
            (PyTorch tensor): logits that 
                correspond to the 
                probability of the image 
                being real
        """
        return self.conv_blocks(img_mid)
    