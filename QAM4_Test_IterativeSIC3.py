import numpy as np 
import time
from CommPy.utils import *
from CommPy.modulation import QAMModem
from initialization import ZF_initial, MMSE_initial
from IterativeSIC import iterative_SIC
import matplotlib.pyplot as plt
from tqdm import tqdm

""" parameters setting """
m = 10                # number of Rx antennas
n = 10                # number of Tx antennas
mod = 4              # modulation order, QAM
SNRdB_range = np.arange(0, 35, 5)
block_size = 10
num_blocks = 10000
H_mean = 0
H_var = 1
num_iter = 5

Tx = QAMModem(mod)
constellation = Tx.constellation
# uniform distribution at first
lld_k_Initial = np.log(1/mod * np.ones(mod)) 

BER_ZF_main = np.zeros(SNRdB_range.shape)
BER_ZFSIC_main = np.zeros(SNRdB_range.shape)
BER_MMSE_main = np.zeros(SNRdB_range.shape)
BER_MMSESIC_main = np.zeros(SNRdB_range.shape)

# main loopS
for dd in range(0, SNRdB_range.shape[0]):
    start_time = time.time()

    # in each SNRdB
    var_noise = np.power(10, -0.1*SNRdB_range[dd])
    BER_ZF_block = np.zeros(num_blocks)
    BER_ZFSIC_block = np.zeros(num_blocks)
    BER_MMSE_block = np.zeros(num_blocks)
    BER_MMSESIC_block = np.zeros(num_blocks)

    for bb in tqdm(range(0, num_blocks)):
        H_real = H_mean + np.sqrt(0.5 * H_var)*np.random.randn(m, n)
        H_imag = H_mean + np.sqrt(0.5 * H_var)*np.random.randn(m, n)
        H = H_real + 1j * H_imag
        bit_width = int(np.sqrt(mod))
        bit_dataset = np.random.randint(0, 2, (bit_width*n, block_size))
        BER_ZF_vector = np.zeros(block_size)
        BER_ZFSIC_vector = np.zeros(block_size)
        BER_MMSE_vector = np.zeros(block_size)
        BER_MMSESIC_vector = np.zeros(block_size)

        for jj in range(0, block_size):
            x_bits = bit_dataset[:, jj]
            x_indices, x_symbols = Tx.modulate(x_bits)
            noise_real = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(m)
            noise_imag = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(m)
            y_symbols = np.matmul(H, x_symbols) + (noise_real + 1j * noise_imag)

            # estimation by zero-forcing
            X_Initial_zf, var_Initial_zf = ZF_initial(H, y_symbols, constellation, lld_k_Initial)
            zf_bits = Tx.demodulate(X_Initial_zf, 'hard')
            BER_ZF_vector[jj] = bit_err_rate(x_bits, zf_bits)

            # estimation by iterative SIC with zf initialization
            xhat_mat, prob = iterative_SIC(X_Initial_zf, var_Initial_zf, var_noise, y_symbols, H, constellation, num_iter)
            xhat_indices = xhat_mat[num_iter-1]
            xhat_bits = np.zeros(x_bits.shape[0], dtype=int)
            for k in range(0, n):
                xhat_bits[k*bit_width : (k+1)*bit_width] = decimal2bitarray(xhat_indices[k], bit_width)
            BER_ZFSIC_vector[jj] = bit_err_rate(x_bits, xhat_bits)

            # estimation by L-MMSE
            lamda = var_noise
            X_Initial_mmse, var_Initial_mmse = MMSE_initial(H, y_symbols, lamda, constellation, lld_k_Initial)
            mmse_bits = Tx.demodulate(X_Initial_mmse, 'hard')
            BER_MMSE_vector[jj] = bit_err_rate(x_bits, mmse_bits)

            # estimation by iterative SIC with mmse initialization
            xhat_mat, prob = iterative_SIC(X_Initial_mmse, var_Initial_mmse, var_noise, y_symbols, H, constellation, num_iter)
            xhat_indices = xhat_mat[num_iter-1]
            xhat_bits = np.zeros(x_bits.shape[0], dtype=int)
            for k in range(0, n):
                xhat_bits[k*bit_width : (k+1)*bit_width] = decimal2bitarray(xhat_indices[k], bit_width)
            BER_MMSESIC_vector[jj] = bit_err_rate(x_bits, xhat_bits)

        BER_ZF_block[bb] = np.mean(BER_ZF_vector)
        BER_ZFSIC_block[bb] = np.mean(BER_ZFSIC_vector)
        BER_MMSE_block[bb] = np.mean(BER_MMSE_vector)
        BER_MMSESIC_block[bb] = np.mean(BER_MMSESIC_vector)

    BER_ZF_main[dd] = np.mean(BER_ZF_block)
    BER_ZFSIC_main[dd] = np.mean(BER_ZFSIC_block)
    BER_MMSE_main[dd] = np.mean(BER_MMSE_block)
    BER_MMSESIC_main[dd] = np.mean(BER_MMSESIC_block)

    print("--- dd=%d --- SNR = %.1f dB --- %s seconds ---" % (dd, SNRdB_range[dd], time.time() - start_time))
    print(" ZF_BER: ")
    print(BER_ZF_main)
    print(" zf-SIC BER: ")
    print(BER_ZFSIC_main)
    print(" MMSE_BER: ")
    print(BER_MMSE_main)
    print(" mmse-SIC BER: ")
    print(BER_MMSESIC_main)

            
# display results
plt.plot(SNRdB_range, BER_ZF_main, '-ro', label='zero-forcing')
plt.plot(SNRdB_range, BER_ZFSIC_main, '-bo', label='Iterative SIC + ZF Init')
plt.plot(SNRdB_range, BER_MMSE_main, '-go', label='MMSE')
plt.plot(SNRdB_range, BER_MMSESIC_main, '-yo', label='Iterative SIC + MMSE Init')
plt.yscale('log')
plt.ylim(1e-5, 1)

plt.xlabel('SNR(dB)')
plt.ylabel('BER')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('./figures_MMSE/Tx10Rx10.png')