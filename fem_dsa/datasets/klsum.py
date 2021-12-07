import os
import math
import torch
import numpy as np
from tqdm import tqdm
import PIL
from torch.utils import data
from DiffNet.gen_input_calc import generate_diffusivity_tensor


class KLSumStochastic(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, filename, domain_size=64, kl_terms=6):
        """
        Initialization
        """
        self.coeffs = np.load(filename)
        self.domain_size = domain_size
        self.kl_terms = kl_terms
        self.dataset = []
        
        print('loading dataset')
        for coeff in tqdm(self.coeffs):
            domain = generate_diffusivity_tensor(coeff, output_size=self.domain_size, n_sum_nu=kl_terms).squeeze()
            # bc1 will be source, u will be set to 1 at these locations
            bc1 = np.zeros_like(domain)
            bc1[:,0] = 1

            # bc2 will be sink, u will be set to 0 at these locations
            bc2 = np.zeros_like(domain)
            bc2[:,-1] = 1

            self.dataset.append(np.array([domain,bc1,bc2]))
        self.dataset = np.array(self.dataset)
        self.n_samples = self.dataset.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = self.dataset[index]
        forcing = np.zeros_like(self.dataset[index][0])
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)




class Dataset(data.Dataset):
    'PyTorch dataset for sampling coefficients'
    def __init__(self, coeff_file, domain_size=64):
        """
        Initialization
        """
        if os.path.exists(coeff_file):
            self.coeff = np.loadtxt(coeff_file, dtype=np.float32)
        else:
            raise FileNotFoundError("Single instance: Wrong path to coefficient file.")
        self.domain_size = domain_size
        self.n_samples = 1000
        self.nu = generate_diffusivity_tensor(self.coeff, output_size=self.domain_size).squeeze()
        # bc1 will be source, u will be set to 1 at these locations
        self.bc1 = np.zeros((domain_size, domain_size))
        self.bc1[:,0] = 1
        # bc2 will be sink, u will be set to 0 at these locations
        self.bc2 = np.zeros((domain_size, domain_size))
        self.bc2[:,-1] = 1
        self.n_samples = 100        

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = np.array([self.nu, self.bc1, self.bc2])
        forcing = np.zeros_like(self.nu)
        return torch.FloatTensor(inputs), torch.FloatTensor(forcing).unsqueeze(0)        
