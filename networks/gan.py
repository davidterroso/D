from numpy import prod
from torch import cat
from torch.nn.functional import pad
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, Dropout2d, Module, LeakyReLU, Sequential, Tanh

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

        self.nfg = 64  # the size of feature map
        self.c = 1  # output channel
        filter_size = 4
        stride_size = 2

        
        self.down_sample_blocks = Sequential(
            Conv2d(self.c * 2, self.nfg * 2, kernel_size=3, stride=1, padding=1, bias=False),  # size
            BatchNorm2d(self.nfg * 2),
            LeakyReLU(0.02, inplace=True),
            Conv2d(self.nfg * 2, self.nfg * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            BatchNorm2d(self.nfg * 2),
            LeakyReLU(0.02, inplace=True),
            Conv2d(self.nfg * 2, self.nfg * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            BatchNorm2d(self.nfg * 4),
            LeakyReLU(0.02, inplace=True),
            Conv2d(self.nfg * 4, self.nfg * 8, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            BatchNorm2d(self.nfg * 8),
            LeakyReLU(0.02, inplace=True)
        )
        
        self.up_sample_block = Sequential(
            ConvTranspose2d(self.nfg * 8, self.nfg * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            BatchNorm2d(self.nfg * 4),
            LeakyReLU(0.02, inplace=True),
            ConvTranspose2d(self.nfg * 4, self.nfg * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            BatchNorm2d(self.nfg * 2),
            LeakyReLU(0.02, inplace=True),
            ConvTranspose2d(self.nfg * 2, self.nfg, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            BatchNorm2d(self.nfg),
            LeakyReLU(0.02, inplace=True),
            ConvTranspose2d(self.nfg, self.c, kernel_size=3, stride=1, padding=1, bias=False),  # size
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
        h0 = int(list(tensor0.size())[2])
        w0 = int(list(tensor0.size())[3])
        h2 = int(list(tensor2.size())[2])
        w2 = int(list(tensor2.size())[3])

        h_padded = False
        w_padded = False
        if (h0 % 32 != 0 or (h0 - w0) < 0):
            pad_h = 32 - (h0 % 32) if (h0 - w0) >= 0 else 32 - (h0 % 32) + (w0 - h0)
            tensor0 = pad(tensor0, (0, 0, 0, pad_h))
            tensor2 = pad(tensor2, (0, 0, 0, pad_h))
            h_padded = True

        if (w0 % 32 != 0 or (h0 - w0) > 0):
            pad_w = 32 - (w0 % 32) if (h0 - w0) <= 0 else 32 - (h0 % 32) + (h0 - w0)
            tensor0 = pad(tensor0, (0, pad_w, 0, 0))
            tensor2 = pad(tensor2, (0, pad_w, 0, 0))
            w_padded = True
        
        out = cat((tensor0, tensor2), 1)
        
        out_down = self.down_sample_blocks(out)
        out_up = self.up_sample_block(out_down)
        
        if h_padded:
            out_up = out_up[:, :, 0:h0, :]
        if w_padded:
            out_up = out_up[:, :, :, 0:w0]
          
        return out_up
    
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
        self.nfg = 64  # the size of feature map
        self.c = 1  # output channel
        
        self.conv_blocks = Sequential(
            # input is c * 64 * 64
            Conv2d(self.c, self.nfg, kernel_size=4, stride=2, padding=1, bias=False),
            LeakyReLU(0.2, inplace=True),
            Dropout2d(0.25),
            # state: nfg * 32 * 32
            Conv2d(self.nfg, self.nfg * 2, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(self.nfg * 2),
            LeakyReLU(0.2, inplace=True),
            Dropout2d(0.25),
            Conv2d(self.nfg * 2, self.nfg * 4, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(self.nfg * 4),
            LeakyReLU(0.2, inplace=True),
            Dropout2d(0.25),
            Conv2d(self.nfg * 4, self.nfg * 8, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(self.nfg * 8),
            LeakyReLU(0.2, inplace=True),
            Dropout2d(0.25),
            Conv2d(self.nfg * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
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
        return self.conv_blocks(x)
    