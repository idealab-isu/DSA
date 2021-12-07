import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=3, encoder_type='convolutional'):
        super(Encoder, self).__init__()
        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim*2, 7),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Downsampling
        for i in range(n_downsample):
            if i <= 3:
                layers += [
                    nn.Conv2d(dim*2*(i+1), dim * (i+2)*2, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(dim * (i+2)*2),
                    nn.ReLU(inplace=True),
                ]
            else:
                layers += [
                    nn.Conv2d(dim*2*(5), dim * (5)*2, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(dim * (5)*2),
                    nn.ReLU(inplace=True),
                ]



        self.model_blocks = nn.Sequential(*layers, nn.Tanh())

    def forward(self, x):
        x = self.model_blocks(x)
        return x



class Decoder(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=3, encoder_type='convolutional', activation='relu'):
        super(Decoder, self).__init__()

        layers = []
        dim = dim 


        # Upsampling
        for i in reversed(range(n_upsample)):
            # print(i)
            if i > 3:
                print('Arjuna')
                layers += [
                    nn.ConvTranspose2d(dim * (5)*2, dim * (5)*2, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(dim * (5)*2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            else:
                layers += [
                    nn.ConvTranspose2d(dim * (i + 2)*2, dim * (i + 1)*2, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(dim * (i + 1)*2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]

        # Output layer
        # layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7)]
        layers += [nn.ReflectionPad2d(4), nn.Conv2d(dim * (i + 1)*2, out_channels, 3), nn.Conv2d(out_channels, out_channels, 7)]

        self.model_blocks = nn.Sequential(*layers)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x):
        # print(x.shape)
        x = self.model_blocks(x)
        #x = self.activation(x)
        return x


class AE(nn.Module):
    """docstring for AE"""
    def __init__(self, in_channels, out_channels, dims=64, n_downsample=4):
        super(AE, self).__init__()
        self.encoder = Encoder(in_channels, dim=dims, n_downsample=n_downsample, encoder_type='regular')
        self.decoder = Decoder(out_channels, dim=dims, n_upsample=n_downsample, activation='relu')

    def forward(self, x):
        code = self.encoder(x)
        out = self.decoder(code)
        return out

class VAE(nn.Module):
    """docstring for AE"""
    def __init__(self, in_channels, out_channels, dims=64, n_downsample=3):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, dim=dims, n_downsample=n_downsample, encoder_type='variational')
        self.decoder = Decoder(out_channels, dim=dims, n_upsample=n_downsample)

    def forward(self, x):
        mu, z = self.encoder(x)
        out = self.decoder(z)
        return out