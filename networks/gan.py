from numpy import prod
from torch import cat
from torch.nn import Conv2d, Flatten, Module, LeakyReLU, Linear, ReLU, Sequential, Sigmoid, Tanh
from networks.unet import DoubleConvolution, DownSample

class Encoder(Module):
    """
    PyTorch Module that encodes the input image, 
    originating the latent state. The encoder 
    corresponds to the U-Net encoder.
    """
    def __init__(self, in_channels: int=3):
        """
        Initiates the Encoder object, which is 
        composed of multiple DownSample blocks 
        used in the U-Net. Each block consists
        of a double convolution and max pooling. 
        
        Args:
            self (Encoder object): the object 
                Encoder that will now be defined
            in_channels (int): Number of channels 
                that are input in this module. 
                Since we are handling three 
                consecutive images, the default 
                number of channels is 3.
        
        Return:
            None
        """
        # Starts the initialization of the 
        # Encoder object
        super(Encoder, self).__init__()
        # Declares the number of input and output channels in 
        # each convolution module. The final 
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        self.bottle_neck = DoubleConvolution(512, 1024)
        # Flattens the PyTorch 
        # tensor to a one 
        # dimension array
        self.flatten = Flatten()
        # Applies a one dimensional linear transformation that transforms the result of the 
        # convolutions into an array with the expected latent shape

    def forward(self, x):
        """
        Forward step of the encoder, 
        returning its result when 
        applied to input image

        Args: 
            self (Encoder object): the 
                Encoder object which is going
                to be performed on input x
            x (PyTorch tensor): input images
                that will be transformed to its
                latent state 

        Return:
            (PyTorch tensor): result of the 
                operations applied to the 
                input x
        """
        # The modules defined in the 
        # initialization of the Encoder 
        # are applied successively 
        x = self.down_convolution_1(x)
        x = self.down_convolution_2(x)
        x = self.down_convolution_3(x)
        x = self.down_convolution_4(x)
        x = self.bottle_neck(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Generator(Module):
    """
    PyTorch Module that generates the 
    intermediate image from the input 
    set of images.
    """
    def __init__(self, latent_dim: int=100, img_shape: tuple=(496,512)):
        """
        Initiates the Generator object, which 
        encodes the image into its latent state
        to which noise is added before being 
        decoded.
        
        Args:
            self (Generator object): the object 
                Generator that will now be defined
            latent_dim (int): size of the latent
                state array. The default value 
                is 100
            img_shape (tuple(int,int)): shape of 
                each input image. The default 
                shape is the shape of the images
                that we are using, which is 
                (496, 512)
        
        Return:
            None
        """
        # Starts the initialization of the 
        # Generator object
        super(Generator, self).__init__()

        # Saves in the object the information of 
        # the image shape, of the encoder, the 
        # linear transformation, and the decoder
        self.img_shape = img_shape
        self.encoder = Encoder()
        # This linear transformation changes the shape of the result of the applied encoding to 
        # the shape expected in the latent dimensions
        self.fc = Linear(1024 * (img_shape[0] / (2 ** 4)) * (img_shape[1] / (2 ** 4)), latent_dim)

        # Decodes the input array into an 
        # image that has the same shape as 
        # the input
        self.decoder = Sequential(
            Linear(latent_dim, 128),
            LeakyReLU(0.2, inplace=True),
            Linear(128, 256),
            LeakyReLU(0.2, inplace=True),
            Linear(256, 512),
            LeakyReLU(0.2, inplace=True),
            Linear(512, 1024),
            ReLU(),
            Linear(1024, prod(img_shape)),
            Tanh()
        )

    def forward(self, img_before, img_after, z):
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
            z (PyTorch tensor): Gaussian
                noise

        Return:
            (PyTorch tensor): result of the 
                operations applied in the 
                forward step to the input x
        """
        # Encodes the previous and following images and 
        # applies the linear information so that it 
        # matches the expected latent shape
        latent_before = self.fc(self.encoder(img_before))
        latent_after = self.fc(self.encoder(img_after))
        # To the average of the latent space of the image before and 
        # the image after (a linear interpolation between latent 
        # spaces), an array of Gaussian noise will be added 
        interpolated = 0.5 * latent_before + 0.5 * latent_after + z 
        # The result of the interpolation 
        # will be decoded
        output = self.decoder(interpolated)
        return output.view(-1, self.img_shape)
    
class Discriminator(Module):
    """
    PyTorch Module that discriminates whether 
    the input set of three consecutive images,
    is real or fake.
    """
    def __init__(self, img_shape):
        """
        Initiates the Discriminator object, which 
        is composed of multiple convolutions and 
        a linear transformation followed by a 
        sigmoid that results in a final binary 
        prediction.
        
        Args:
            self (Discriminator object): the object 
                Discriminator that will now be defined
            img_shape (tuple(int,int)): shape of 
                each input image. The default 
                shape is the shape of the images
                that we are using, which is 
                (496, 512)
        
        Return:
            None
        """
        # Starts the initialization of the 
        # Discriminator object
        super(Discriminator, self).__init__()

        # Defines a sequence of two-dimensional convolutions 
        # and ReLU activations, that are applied to the input 
        # image before converting it, through a linear 
        # transformation to a single value, to which is 
        # applied a sigmoid, resulting in a final prediction
        self.model = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            LeakyReLU(0.2, inplace=True),
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            LeakyReLU(0.2, inplace=True),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            LeakyReLU(0.2, inplace=True),
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            LeakyReLU(0.2, inplace=True),
            Flatten(),
            Linear(512 * int(img_shape[0] / (2 ** 4)) * int(img_shape[1] / (2 ** 4)), 1),
            Sigmoid()
        )

    def forward(self, img_before, img_mid, 
                img_after):
        """
        Forward step of the discriminator, 
        that returns the probability of the 
        image being real

        Args: 
            self (Generator object): the 
                Generator object
            img_before (PyTorch tensor): 
                image before the one we 
                are looking to classify
            img_mid (PyTorch): image we
                are looking to classify
            img_after (PyTorch tensor): 
                image before the one we 
                are looking to classify
            z (PyTorch tensor): Gaussian
                noise

        Return:
            (PyTorch tensor): probability 
                of the image being real
        """
        # Concatenates the images along the 
        # now first dimension
        x = cat([img_before, img_mid, img_after], dim=1)
        return self.model(x)
    