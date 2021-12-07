import torch
import numpy as np
from torch import nn
from torch.autograd import grad
from torch.nn.modules.utils import _pair

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, kernel_size=4, stride=2, padding=1, dilation=1):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size, stride, padding, dilation, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GoodEncoder(nn.Module):
    def __init__(self, in_channels=1, in_dim=64, lowest_dim=8, filters=4):
        super(GoodEncoder, self).__init__()
        self.in_channels = in_channels    
        assert in_dim > 8
        resize_factor = int(np.floor(np.log2(in_dim/lowest_dim)))
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        low_log2 = int(np.log2(lowest_dim))

        # First layer
        cur_dim = in_dim
        target_dim = 2**(resize_factor + low_log2 - 1)
        kernel_size = cur_dim - 2*(target_dim - 1) + 2
        self.downs.append(UNetDown(in_channels, filters, stride=2, kernel_size=kernel_size, padding=1, dilation=1))
        
        # Rest of down layers
        cur_dim = 2**(resize_factor + low_log2 - 1)
        for layer_idx in range(resize_factor - 2):
            # self.downs.append(UNetDown(filters, filters*2, dropout=0.1+0.4*(layer_idx/resize_factor)))
            self.downs.append(UNetDown(filters, filters*2, normalize=False))
            filters = 2*filters
            cur_dim = int(cur_dim/2)

        # Last down layer
        self.downs.append(UNetDown(filters, filters, normalize=False))
        assert (cur_dim/2) == lowest_dim

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        downs = []
        for down in self.downs:
            x = down(x)
            downs.append(x)
        return downs
        

class GoodDecoder(nn.Module):
    def __init__(self, out_channels=1, out_dim=64, lowest_dim=4, filters=32):
        super(GoodDecoder, self).__init__()
        self.out_channels = out_channels
        self.ups = nn.ModuleList()
        resize_factor = int(np.floor(np.log2(out_dim/lowest_dim)))
        filters = int(filters*2**(resize_factor-2))
        cur_dim = int(lowest_dim)
        # First up layer
        self.ups.append(UNetUp(filters, filters, dropout=0.5))
        cur_dim = int(cur_dim*2)
        
        # Rest of up layers
        for layer_idx in reversed(range(resize_factor)):
            self.ups.append(UNetUp(2*filters, int(filters/2)))
            filters = int(filters/2)
            cur_dim = int(cur_dim*2)
        
        # Last up layer
        filters = 8*filters
        cur_dim = int(cur_dim/4)
        kernel_size = out_dim - (cur_dim - 1)*2 + 2
        self.final = nn.ModuleList()
        if kernel_size < 10:
            self.final.append(nn.ConvTranspose2d(filters, out_channels, kernel_size, padding=1, stride=2, dilation=1, output_padding=0))    
        else:
            num_int_layers = filters - 2
            s_dim = cur_dim
            for l_idx in range(num_int_layers):
                int_dim = int((l_idx+1)*(out_dim-s_dim)/(num_int_layers)) + s_dim
                kernel_size = int_dim - (cur_dim - 1) + 2
                conv_layer = nn.ConvTranspose2d(filters, filters-1, kernel_size, padding=1, stride=1, dilation=1, output_padding=0)
                self.final.append(conv_layer)
                filters = filters - 1
                cur_dim = int_dim
            kernel_size = out_dim - (int_dim - 1) + 2
            conv_layer = nn.ConvTranspose2d(filters, out_channels, kernel_size, padding=1, stride=1, dilation=1, output_padding=0)
            self.final.append(conv_layer)

        activation = nn.Sigmoid()
        self.final.append(activation)

    def forward(self, downs):
        downs = downs[::-1]
        u = downs[0]
        for idx in range(len(downs)-1):
            u = self.ups[idx](u, downs[idx+1])
        
        out = u
        for f in self.final:
            out = f(out)
        return out

    
class GoodNetwork(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, in_dim=64, out_dim=64, lowest_dim=4, filters=16):
        super(GoodNetwork, self).__init__()
        assert in_dim > 8
        self.encoder = GoodEncoder(in_channels=in_channels, in_dim=in_dim, lowest_dim=lowest_dim, filters=filters)
        self.decoder = GoodDecoder(out_channels=out_channels, out_dim=out_dim, lowest_dim=lowest_dim, filters=filters)


    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        downs = self.encoder(x)
        out = self.decoder(downs)
        return out