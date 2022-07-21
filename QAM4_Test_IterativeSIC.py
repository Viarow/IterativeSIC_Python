import numpy as np 
import time
from CommPy.utils import *
from CommPy.modulation import QAMModem
from EstimateInitial import estimate_initial
from IterativeSIC import iterative_SIC
import matplotlib.pyplot as plt
from tqdm import tqdm

""" parameters setting """
m = 3                # number of Rx antennas
n = 3                # number of Tx antennas
mod = 4              # modulation order, QAM
SNRdB_range = np.arange(0, 35, 5)
block_size = 10
num_blocks = 10
H_mean = 0
H_var = 1
num_iter = 5

Tx = QAMModem(mod)
constellation = Tx.constellation
# uniform distribution at first
lld_k_Initial = np.log(1/mod * np.ones(mod)) 

BER_ZF_main = np.zeros(SNRdB_range.shape)
BER_SIC_main = np.zeros(SNRdB_range.shape)

# main loopS
for dd in range(0, SNRdB_range.shape[0]):
    start_time = time.time()

    # in each SNRdB
    var_noise = np.power(10, -0.1*SNRdB_range[dd])
    BER_ZF_block = np.zeros(num_blocks)
    BER_SIC_block = np.zeros(num_blocks)

    for bb in tqdm(range(0, num_blocks)):
        H_real = H_mean + np.sqrt(0.5 * H_var)*np.random.randn(m, n)
        H_imag = H_mean + np.sqrt(0.5 * H_var)*np.random.randn(m, n)
        H = H_real + 1j * H_imag
        bit_width = int(np.sqrt(mod))
        bit_dataset = np.random.randint(0, 2, (bit_width*n, block_size))
        BER_ZF_vector = np.zeros(block_size)
        BER_SIC_vector = np.zeros(block_size)

        for jj in range(0, block_size):
            x_bits = bit_dataset[:, jj]
            x_indices, x_symbols = Tx.modulate(x_bits)
            noise_real = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(m)
            noise_imag = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(m)
            y_symbols = np.matmul(H, x_symbols) + (noise_real + 1j * noise_imag)

            # estimation by zero-forcing
            X_Initial, var_Initial = estimate_initial(H, y_symbols, constellation, lld_k_Initial)
            zf_bits = Tx.demodulate(X_Initial, 'hard')
            BER_ZF_vector[jj] = bit_err_rate(x_bits, zf_bits)

            # estimation by iterative SIC
            xhat_mat, prob = iterative_SIC(X_Initial, var_Initial, var_noise, y_symbols, H, constellation, num_iter)
            xhat_indices = xhat_mat[num_iter-1]
            xhat_bits = np.zeros(x_bits.shape[0], dtype=int)
            for k in range(0, n):
                xhat_bits[k*bit_width : (k+1)*bit_width] = decimal2bitarray(xhat_indices[k], bit_width)
            BER_SIC_vector[jj] = bit_err_rate(x_bits, xhat_bits)

        BER_ZF_block[bb] = np.mean(BER_ZF_vector)
        BER_SIC_block[bb] = np.mean(BER_SIC_vector)

    BER_ZF_main[dd] = np.mean(BER_ZF_block)
    BER_SIC_main[dd] = np.mean(BER_SIC_block)

    print("--- dd=%d --- SNR = %.1f dB --- %s seconds ---" % (dd, SNRdB_range[dd], time.time() - start_time))
    print(" ZF_BER: ")
    print(BER_ZF_main)
    print(" SIC BER: ")
    print(BER_SIC_main)

            
# display results
plt.plot(SNRdB_range, BER_ZF_main, '-ro', label='zero-forcing')
plt.plot(SNRdB_range, BER_SIC_main, '-bo', label='Iterative SIC')
plt.yscale('log')
plt.ylim(1e-5, 1)

plt.xlabel('SNR(dB)')
plt.ylabel('BER')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('./figures/Tx3Rx3.png')