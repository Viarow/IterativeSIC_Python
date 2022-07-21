import torch
from torch import nn
import numpy as np 
from network import InitNet
import time
from CommPy.utils import *
from CommPy.modulation import QAMModem
from EstimateInitial import estimate_initial
from IterativeSIC import iterative_SIC
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_args():

    """ parameters setting """
    params = {
        'm' : 2,
        'n' : 2,
        'mod': 4,
        'H_mean': 0,
        'H_var': 1,
        'block_size': 100,
        'num_blocks': 1000,
        'num_iter': 5,
        'size': 20000,
        'batch_size': 16,
        'epochs': 10,
        'learning_rate': 1e-3
    }
    SNRdB_range = np.arange(0, 35, 5)

    return params, SNRdB_range


def plot_curves(SNRdB_range, BER_data, fig_path):
    plt.plot(SNRdB_range, BER_data['ZF_init'], '-ro', label='ZF Init')
    plt.plot(SNRdB_range, BER_data['NN_init'], '-bo', label='NN Init')
    plt.yscale('log')
    plt.ylim(1e-5, 1)

    plt.xlabel('SNR(dB)')
    plt.ylabel('BER')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(fig_path)


def apply_InitNet(model, H, y, cuda=False):

    # x_cat = np.concatenate((np.real(x), np.imag(x)))
    H_flat = np.reshape(H, (H.shape[0]*H.shape[1], ))
    y_cat = np.concatenate((np.real(y), np.imag(y)))
    H_cat = np.concatenate((np.real(H_flat), np.imag(H_flat)))
    y_H = np.concatenate((y_cat, H_cat))
    y_H = torch.from_numpy(y_H).float()
    
    if cuda:
        y_H = y_H.cuda()
        xhat_cat = model(y_H)
        xhat_cat = xhat_cat.cpu()
    else:
        xhat_cat = model(y_H)
    
    xhat_real = xhat_cat[0 : H.shape[1]].detach().numpy()
    xhat_imag = xhat_cat[H.shape[1] : 2*H.shape[1]].detach().numpy()
    xhat = xhat_real + 1j * xhat_imag

    return xhat


def get_BER(xhat_indices, x_bits, n, bit_width):
    xhat_bits = np.zeros(x_bits.shape[0], dtype=int)
    for k in range(0, n):
        xhat_bits[k*bit_width : (k+1)*bit_width] = decimal2bitarray(xhat_indices[k], bit_width)
    BER = bit_err_rate(x_bits, xhat_bits)
    return BER
        

def main_loops(params, SNRdB_range, ckpt_path, cuda=False):
    
    m = params['m']
    n = params['n']
    mod = params['mod']
    num_blocks = params['num_blocks']
    block_size = params['block_size']
    H_mean = params['H_mean']
    H_var = params['H_var']
    num_iter = params['num_iter']
    
    Tx = QAMModem(mod)
    constellation = Tx.constellation
    # uniform distribution at first
    lld_k_Initial = np.log(1/mod * np.ones(mod)) 

    # build InitNet
    model = InitNet(params['m'], params['n'])
    if cuda:
        model = model.cuda()
    model.load_state_dict(torch.load(ckpt_path))

    # ZF init and NN init respectively
    BER_ZF_main = np.zeros(SNRdB_range.shape)
    BER_NN_main = np.zeros(SNRdB_range.shape)
    
    for dd in range(0, SNRdB_range.shape[0]):
        start_time = time.time()

        var_noise = np.power(10, -0.1*SNRdB_range[dd])
        BER_ZF_block = np.zeros(num_blocks)
        BER_NN_block = np.zeros(num_blocks)

        for bb in tqdm(range(0, num_blocks)):
            H_real = H_mean + np.sqrt(0.5 * H_var)*np.random.randn(m, n)
            H_imag = H_mean + np.sqrt(0.5 * H_var)*np.random.randn(m, n)
            H = H_real + 1j * H_imag
            bit_width = int(np.sqrt(params['mod']))
            bit_dataset = np.random.randint(0, 2, (bit_width*n, block_size))
            BER_ZF_vector = np.zeros(block_size)
            BER_NN_vector = np.zeros(block_size)

            for jj in range(0, block_size):
                x_bits = bit_dataset[:, jj]
                x_indices, x_symbols = Tx.modulate(x_bits)
                noise_real = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(m)
                noise_imag = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(m)
                y_symbols = np.matmul(H, x_symbols) + (noise_real + 1j * noise_imag)

                # Initialization by zero-forcing
                X_Initial_ZF, var_Initial_ZF = estimate_initial(H, y_symbols, constellation, lld_k_Initial)

                # Initilization by NN
                model.eval()
                X_Initial_NN = apply_InitNet(model, H, y_symbols, cuda)
                var_Initial_NN = np.zeros(n)
                for k in range(0, n):
                    var_Initial_NN[k] = np.sum(np.square(np.abs(constellation-X_Initial_NN[k])) * np.exp(lld_k_Initial)) / np.sum(np.exp(lld_k_Initial))

                # Estimation by IterativeSIC
                xhat_ZF, prob_ZF = iterative_SIC(X_Initial_ZF, var_Initial_ZF, var_noise, y_symbols, H, constellation, num_iter)
                BER_ZF_vector[jj] = get_BER(xhat_ZF, x_bits, n, bit_width)
                
                xhat_NN, prob_NN = iterative_SIC(X_Initial_NN, var_Initial_NN, var_noise, y_symbols, H, constellation, num_iter)
                BER_NN_vector[jj] = get_BER(xhat_NN, x_bits, n, bit_width)

            BER_ZF_block[bb] = np.mean(BER_ZF_vector)
            BER_NN_block[bb] = np.mean(BER_NN_vector)
        
        BER_ZF_main[dd] = np.mean(BER_ZF_block)
        BER_NN_main[dd] = np.mean(BER_NN_block)

        print("--- dd=%d --- SNR = %.1f dB --- %s seconds ---" % (dd, SNRdB_range[dd], time.time() - start_time))
        print(" with zero-forcing initialization: ")
        print(BER_ZF_main)
        print(" with neural network initialization: ")
        print(BER_NN_main)

    BER_data = {'ZF_init': BER_ZF_main, 'NN_init': BER_NN_main}
    return BER_data


if __name__ == '__main__':

    params, SNRdB_range = get_args()
    ckpt_path = './workspaces/Tx2Rx2_CUDA.pt'
    BER_data = main_loops(params, SNRdB_range, ckpt_path, True)
    fig_path = './figures/NNInit_Tx2Rx2.png'
    plot_curves(SNRdB_range, BER_data, fig_path)