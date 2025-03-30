from torch.nn import BatchNorm1d, Module, LeakyReLU, Linear, Sequential, Sigmoid, Tanh
from numpy import prod

class Generator(Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers=[Linear(in_feat, out_feat)]
            if normalize:
                layers.append(BatchNorm1d(out_feat, 0.8))
            layers.append(LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = Sequential(
            *block(latent_dim, 128, normalize=False)
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            Linear(1024, int(prod(img_shape))),
            Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
    
class Discriminator(Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = Sequential(
            Linear(int(prod(img_shape)), 512),
            LeakyReLU(0.2, inplace=True),
            Linear(512, 256),
            LeakyReLU(0.2, inplace=True),
            Linear(256, 1),
            Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
    