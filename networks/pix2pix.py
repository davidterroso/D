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

    def __init__(self, input_nc, output_nc, num_downs=6, ngf=64, norm_layer=BatchNorm2d, use_dropout=False):
        """
        Initiates the generator object that will receive multiple
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 6,
                                image of size 64x64 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Pix2PixGenerator, self).__init__()
        # construct unet structure
        unet_block = SkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = SkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = SkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = SkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = SkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = SkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

    def get_net_params(self):
        for name, params in self.named_parameters():
            print(name, params.size())


class SkipConnectionBlock(Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (SkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(SkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == partial:
            use_bias = norm_layer.func == InstanceNorm2d
        else:
            use_bias = norm_layer == InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
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

        self.model = Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return cat([x, self.model(x)], 1)  # @UndefinedVariable
        
class Pix2PixDiscriminator(Module):
    """
    Defines a PatchGAN discriminator
    Require two images as input: both input and output of the generator
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Pix2PixDiscriminator, self).__init__()
        if type(norm_layer) == partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == InstanceNorm2d
        else:
            use_bias = norm_layer == InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [Conv2d(input_nc * 2, ndf, kernel_size=kw, stride=2, padding=padw), LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            LeakyReLU(0.2, True)
        ]

        sequence += [Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=0)]  # output 1 channel prediction map
        sequence += [Sigmoid()]
        self.model = Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
    def get_net_params(self):
        for name, params in self.named_parameters():
            print(name, params.size())
