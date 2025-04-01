from torch.nn import Module, LeakyReLU, Linear, ReLU, Sequential, Sigmoid, Tanh
from numpy import prod
from torchvision import models

class Generator(Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        self.img_shape = img_shape
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = Linear(512, latent_dim)

        self.decoder = Sequential(
            Linear(latent_dim, 128),
            LeakyReLU(0.2, inplace=True),
            Linear(128, 256),
            LeakyReLU(0.2, inplace=True),
            Linear(256, 512),
            LeakyReLU(0.2, inplace=True),
            Linear(512, 1024),
            ReLU(),
            Linear(1024, 3 * prod(img_shape)),
            Tanh()
        )

    def forward(self, img_before, img_after, z):
        latent_before = self.encoder(img_before)
        latent_after = self.encoder(img_after)
        interpolated = 0.5 * latent_before + 0.5 * latent_after + z 
        output = self.decoder(interpolated)
        return output.view(-1, 3, self.img_shape)
    
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
    