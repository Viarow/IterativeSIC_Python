import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from CommPy.modulation import QAMModem


class QAM_SingleDataset(Dataset):

    def __init__(self, params, SNRdB, transform=None):
        """ sample at one SNR level
            Rayleigh Fading   
        """
        self.m = params['m']
        self.n = params['n']
        self.mod = params['mod']
        self.Tx = QAMModem(params['mod'])
        self.H_mean = params['H_mean']
        self.H_var = params['H_var']
        self.size = params['size']
        self.SNRdB = SNRdB

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # totally random
        H_real = self.H_mean + np.sqrt(0.5 * self.H_var) * torch.randn(self.m, self.n)
        H_imag = self.H_mean + np.sqrt(0.5 * self.H_var) * torch.randn(self.m, self.n)
        H = torch.complex(H_real, H_imag)
        
        bit_width = int(np.sqrt(self.mod))
        x_bits = np.random.randint(0, 2, (bit_width*self.n, ))
        x_indices, x_symbols = self.Tx.modulate(x_bits)
        x_real = torch.from_numpy(np.real(x_symbols)).float()
        x_imag = torch.from_numpy(np.imag(x_symbols)).float()
        x_symbols = torch.complex(x_real, x_imag)
        
        var_noise = np.power(10, -0.1*self.SNRdB)
        noise_real = 0. + np.sqrt(0.5 * var_noise)*torch.randn(self.m)
        noise_imag = 0. + np.sqrt(0.5 * var_noise)*torch.randn(self.m)
        y_symbols = torch.matmul(H, x_symbols) + torch.complex(noise_real, noise_imag)

        x_cat = torch.cat((x_symbols.real, x_symbols.imag))
        H_flat = torch.reshape(H, (self.m * self.n, ))
        y_cat = torch.cat((y_symbols.real, y_symbols.imag))
        H_cat = torch.cat((H_flat.real, H_flat.imag))
        # y_cat: 2m  H_cat: 2 * m * n, y_H: 2m * (1+n)
        y_H = torch.cat((y_cat, H_cat))

        sample = {'input': y_H, 'label': x_cat}
        return sample


class QAM_MixedDateset(Dataset):

    def __init__(self, params, SNRdB_range, transform=None):
        """ sample over a range of SNR levels
            Rayleigh Fading   
        """
        self.m = params['m']
        self.n = params['n']
        self.mod = params['mod']
        self.Tx = QAMModem(params['mod'])
        self.H_mean = params['H_mean']
        self.H_var = params['H_var']
        self.num_blocks = params['num_blocks']
        # size for each SNR
        self.size = params['size']
        self.SNRdB_range = torch.from_numpy(SNRdB_range)
        self.SNR_list = torch.from_numpy(SNRdB_range).repeat(params['size'])

    def __len__(self):
        return self.SNRdB_range.shape[0] * self.size

    def __getitem__(self, idx):
        H_real = self.H_mean + np.sqrt(0.5 * self.H_var) * torch.randn(self.m, self.n)
        H_imag = self.H_mean + np.sqrt(0.5 * self.H_var) * torch.randn(self.m, self.n)
        H = torch.complex(H_real, H_imag)

        bit_width = int(np.sqrt(self.mod))
        x_bits = np.random.randint(0, 2, (bit_width*self.n, ))
        x_indices, x_symbols = self.Tx.modulate(x_bits)
        x_real = torch.from_numpy(np.real(x_symbols)).float()
        x_imag = torch.from_numpy(np.imag(x_symbols)).float()
        x_symbols = torch.complex(x_real, x_imag)

        var_noise = np.power(10, -0.1*self.SNR_list[idx])
        noise_real = 0. + np.sqrt(0.5 * var_noise)*torch.randn(self.m)
        noise_imag = 0. + np.sqrt(0.5 * var_noise)*torch.randn(self.m)
        y_symbols = torch.matmul(H, x_symbols) + torch.complex(noise_real, noise_imag)

        x_cat = torch.cat((x_symbols.real, x_symbols.imag))
        H_flat = torch.reshape(H, (self.m * self.n, ))
        y_cat = torch.cat((y_symbols.real, y_symbols.imag))
        H_cat = torch.cat((H_flat.real, H_flat.imag))
        # y_cat: 2m  H_cat: 2 * m * n, y_H: 2m * (1+n)
        y_H = torch.cat((y_cat, H_cat))

        sample = {'input': y_H, 'label': x_cat}
        return sample
