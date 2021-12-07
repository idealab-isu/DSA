import os
import math
import torch
import numpy as np
from torch.utils import data

class Rectangle(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.ones((domain_size, domain_size))
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        self.bc1[0,:] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[-1,:] = 1
        self.n_samples = 6000
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = np.zeros_like(self.domain)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)



class RectangleManufactured(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.ones((domain_size, domain_size))
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[-1,:] = 1
        self.bc2[0,:] = 1
        self.bc2[:,0] = 1
        self.bc2[:,-1] = 1
        self.n_samples = 100
        x = np.linspace(0,1,domain_size)
        y = np.linspace(0,1,domain_size)
        xx, yy = np.meshgrid(x,y)
        self.forcing = 2. * math.pi**2 * np.sin(math.pi * xx) * np.sin(math.pi * yy)
        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = self.forcing
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)



class RectangleManufacturedNonZeroBC(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, domain_size=64):
        """
        Initialization
        """
        self.domain = np.ones((domain_size, domain_size))
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        self.bc1[:,0] = 1
        self.bc1[:,-1] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[-1,:] = 1
        self.bc2[0,:] = 1
        self.n_samples = 100
        x = np.linspace(0,1,domain_size)
        y = np.linspace(0,1,domain_size)
        xx, yy = np.meshgrid(x,y)
        self.xx = xx
        self.yy = yy
        self.om = np.pi
        self.u_exact = np.exp(-self.om*xx)*np.sin(self.om*yy)
        self.forcing = np.zeros_like(xx)


    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.domain, self.bc1, self.bc2])
        forcing = self.forcing
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)
