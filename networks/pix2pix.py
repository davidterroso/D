from functools import partial
from torch import cat
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d,\
                     Dropout, InstanceNorm2d, Module,\
                     LeakyReLU, ReLU, Sequential, Sigmoid, Tanh

class Pix2PixGenerator(Module):
    """
    U-Net inspired generator that is 
    composed of multiple encoding 
    decoding blocks

	The orginal implementation is 
    https://github.com/phillipi/pix2pix
	Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu
	"""

    def __init__(self, input_nc, output_nc, num_downs=6, 
                 ngf=64, norm_layer=BatchNorm2d, 
                 use_dropout=False):
        """
        Initiates the generator object, which is composed 
        by multiple encoding and decoding blocks. The 
        network is constructed from the innermost layer 
        to the outermost layer through a recursive process.
        It also adds the option of batch normalization 
        and dropout

        Args:
            self (PyTorch Module): the PyTorch Module 
                class that will be initiated
            input_nc (int): number of channels in the 
                input images
            output_nc (int): number of channels in the 
                output images
            num_downs (int): the number of downsamplings 
                in the network            
            ngf (int): number of filters in the last 
                convolutional layer
            norm_layer (PyTorch Module): normalization 
                layer
            use_dropout (bool): argument that indicates 
                whether dropout will be used or not
        
        Returns:
            None
        """
        # Initiates the Pix2PixGenerator object
        super(Pix2PixGenerator, self).__init__()
        # Starts by creating the innermost layer, which corresponds to the one in the bottle neck
        unet_block = SkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        # Creates intermediate layers
        for i in range(num_downs - 5):
            unet_block = SkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # Creates the multiple encoding blocks and decoding blocks
        unet_block = SkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = SkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = SkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # Finally creates the final model
        # While this is the last block, it calls the previously created block, the same way all the the previous blocks do
        # This is why the process is labeled as recursive, since each block calls the previous block until the last block is created,
        # composing the model
        self.model = SkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        """
        Calls the forward step 
        of the U-Net using the 
        input which is the image 
        that will be transformed
        
        Args:
            self (PyTorch Module):
                the initiated 
                Pix2PixGenerator 
                PyTorch Module
            input (PyTorch tensor):
                image to transform 
                as a PyTorch tensor
        
        Returns:
            (PyTorch tensor): the 
                result output from 
                the model for the 
                given input 
        """
        return self.model(input)

    def get_net_params(self):
        """
        Outputs the total number of trainable 
        parameters of the Pix2PixGenerator for each 
        layer

        Args:
            self (PyTorch Module): The initiated 
            Pix2PixGenerator PyTorch Module

        Returns:
            None
        """
        # Iterates through all the named parameters 
        # and prints the total number of trainable
        # parameters in each layer
        for name, params in self.named_parameters():
            print(name, params.size())


class SkipConnectionBlock(Module):
    """
    Class that defines one downsampling and upsampling 
    modules as well as the skip connection between them, 
    for a given depth 
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, 
                 innermost=False, norm_layer=BatchNorm2d, 
                 use_dropout=False):
        """
        Constructs each U-Net layer of downsampling and 
        upsampling block, as well as the skip connection 
        between them

        Args:
            outer_nc (int): number of filters in the 
                outer convolutional layer
            inner_nc (int): number of filters in the 
                inner convolutional layer
            input_nc (int): number of channels in the 
                input, which can be images or features
            submodule (SkipConnectionBlock Module): 
                previously defined SkipConnectionBlock
            outermost (bool): bool that indicates if 
                this module is the outermost module
            innermost (bool): bool that indicates if 
                this module is the innermost module
            norm_layer (PyTorch Module): PyTorch Module 
                that will be used as the normalization 
                layer 
            use_dropout (bool): bool that indicates if 
                dropout layers will be used

        Returns:
            None
        """
        # Initiates the SkipConnectionBlock as a 
        # PyTorch Module 
        super(SkipConnectionBlock, self).__init__()
        # Saves in the model if it 
        # is the outermost layer
        self.outermost = outermost
        # Determines whether a bias will be used or not
        # The bias is generally unnecessary when 
        # normalization is performed
        if type(norm_layer) == partial:
            use_bias = norm_layer.func == InstanceNorm2d
        else:
            use_bias = norm_layer == InstanceNorm2d
        # In case the number of input channels is None, 
        # it is assumed that it is the same as the output 
        # channels
        if input_nc is None:
            input_nc = outer_nc
        # Creates a downconvolution with the number of 
        # channels and filters that will be used
        downconv = Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        # Creates the ReLU operation as 
        # inplace, meaning that it will 
        # be applied directly in the 
        # variable without returning something
        downrelu = LeakyReLU(0.2, True)
        # Initiates the normalization 
        # in the downsampling block
        downnorm = norm_layer(inner_nc)
        # Creates the ReLU operation 
        # that will be used in the 
        # upsampling, that will also 
        # be applied inplace
        uprelu = ReLU(True)
        # Initiates the normalization 
        # in the upsampling block
        upnorm = norm_layer(outer_nc)

        # If the layer is the outermost it calls the downsampling,
        # the previously defined module (which is inner than the 
        # current module and calls the other submodules) followed 
        # by the upsampling and activation, which is the hyperbolic 
        # tangent
        if outermost:
            upconv = ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, Tanh()]
            model = down + [submodule] + up
        # If the layer is the innermost it calls the downsampling,
        # followed by the upsampling block. There is no submodule 
        # that is located closer to the center than this one, thus
        # it is not being called
        elif innermost:
            upconv = ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        # For every other layer that is not the innermost or the 
        # outermost, downsampling, the previously defined submodule 
        # (located closer to the center), and the upsampling block
        # is called. The difference between this and the one in the 
        # outermost is the presence of the dropout layer and the 
        # optional use of bias, which is mandatory in the outermost
        else:
            upconv = ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [Dropout(0.5)]
            else:
                model = down + [submodule] + up

        # The model is built as the 
        # sequence of the multiple blocks
        self.model = Sequential(*model)

    def forward(self, x):
        """
        Forward step of the SkipConnectionModule 
        that performs the skip connection 
        between the result of the submodules 
        in case it is not the outermost module

        Args:
            self (SkipConnectionBlock Module):
                the initiated SkipConnectionBlock
                PyTorch Module
            x (PyTorch tensor): input of the
                SkipConnectionBlock which can be 
                either an image or a feature map 
        """
        # In case it is the outermost block, it 
        # just returns the result from the 
        # up-sampling layer
        if self.outermost:
            return self.model(x)
        # If it is not the outermost block, it
        # concatenates the input of the block
        # with its output, performing skip 
        # connection
        else:
            return cat([x, self.model(x)], 1)
        
class Pix2PixDiscriminator(Module):
    """
    Defines the discriminator that is used in the Pix2Pix network. This 
    module receives as input two images (one is real while the other may 
    be real or fake) and classifies the pair either as real or fake
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=BatchNorm2d):
        """
        Initiates the discriminator object as a PyTorch Module. This module 
        consists of a number of convolutional layers that depends of the 
        given argument

        Args:
            input_nc (int): number of channels in the input images
            ndf (int): number of filters in the last convolutional layer
            n_layers (int): number of convolutional layers in the 
                discriminator
            norm_layer (PyTorch Module): PyTorch Module that will be used 
                as the normalization layer

        Returns:
            None
        """
        # Constructs the Pix2PixDiscriminator as a PyTorch Module
        super(Pix2PixDiscriminator, self).__init__()

        # Determines whether a bias will be used or not
        # The bias is generally unnecessary when 
        # normalization is performed
        if type(norm_layer) == partial: 
            use_bias = norm_layer.func == InstanceNorm2d
        else:
            use_bias = norm_layer == InstanceNorm2d

        # Defines the shape of the kernel 
        # and the size of the padding
        kw = 4
        padw = 1
        # Defines the sequence of operations that compose the discriminator, beginning with a convolution 
        # and a Leaky ReLU as the activation function 
        sequence = [Conv2d(input_nc * 2, ndf, kernel_size=kw, stride=2, padding=padw), LeakyReLU(0.2, True)]
        # Defines the multiplier of the 
        # number of filters in the current 
        # and the previous convolutional layer
        nf_mult = 1
        nf_mult_prev = 1
        # Loops through the desired 
        # number of layers and creates 
        # the corresponding convolutions
        for n in range(1, n_layers):
            # The multiplier of previous
            # filters is updated to the
            # current value assigned to 
            # the multiplier of filters 
            nf_mult_prev = nf_mult
            # The current multiplier 
            # is defined as the 2 raised 
            # to depth of the layer or 8
            # (whichever is the lowest)
            nf_mult = min(2 ** n, 8)
            # Appends to the sequence of operations another convolution, a layer normalization, and a leaky ReLU 
            # as activation, considering the arguments given
            sequence += [
                Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                LeakyReLU(0.2, True)
            ]

        # Updates the previous 
        # number of filters 
        # multiplier to its 
        # current value
        nf_mult_prev = nf_mult
        # The current multiplier 
        # is defined as the 2 raised 
        # to total number of layers 
        # or 8 (whichever is the lowest)
        nf_mult = min(2 ** n_layers, 8)
        # Appends to the sequence of operations another convolution, a layer normalization, and a leaky ReLU 
        # as activation, considering the arguments given
        sequence += [
            Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            LeakyReLU(0.2, True)
        ]

        # Appends to the sequence of operations the final convolution, that ends 
        # with a number of channels equal to 1, followed by a sigmoid
        sequence += [Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=0)]
        sequence += [Sigmoid()]

        # Converts all the models into a 
        # single Sequential Module object 
        # that composes our model
        self.model = Sequential(*sequence)

    def forward(self, input):
        """
        Forward step of the Pix2PixDiscriminator 
        that performs the convolutions on the 
        input resulting in the predicted labels

        Args:
            self (Pix2PixDiscriminator Module):
                the initiated Pix2PixDiscriminator
                PyTorch Module
            x (PyTorch tensor): input image of the
                Pix2PixDiscriminator
        """
        return self.model(input)
    
    def get_net_params(self):
        """
        Outputs the total number of trainable 
        parameters of the Pix2PixDiscriminator 
        for each layer

        Args:
            self (PyTorch Module): The initiated 
            Pix2PixDiscriminator PyTorch Module

        Returns:
            None
        """
        # Iterates through all the named parameters 
        # and prints the total number of trainable
        # parameters in each layer
        for name, params in self.named_parameters():
            print(name, params.size())
