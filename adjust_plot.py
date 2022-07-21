import numpy as np 
import time
import matplotlib.pyplot as plt

SNRdB_range = np.arange(0, 35, 5)
BER_ZF_main = np.array([0.146718, 0.063001, 0.022953, 0.007397, 0.002696, 0.000663, 0.000248])
BER_ZFSIC_main = np.array([2.207e-03, 6.140e-04, 1.500e-04, 8.400e-05, 4.800e-05, 1.000e-05, 6.000e-06])
BER_MMSE_main = np.array([1.8234e-02, 4.2390e-03, 1.0100e-03, 3.2300e-04, 9.1000e-05, 2.6000e-05, 9.0000e-06])
BER_MMSESIC_main = np.array([3.43e-04, 1.00e-05, 2.00e-06, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00])

# display results
plt.plot(SNRdB_range, BER_ZF_main, '-ro', label='zero-forcing')
plt.plot(SNRdB_range, BER_ZFSIC_main, '-bo', label='Iterative SIC + ZF Init')
plt.plot(SNRdB_range, BER_MMSE_main, '-go', label='MMSE')
plt.plot(SNRdB_range, BER_MMSESIC_main, '-yo', label='Iterative SIC + MMSE Init')
plt.yscale('log')
plt.ylim(1e-6, 1)

plt.xlabel('SNR(dB)')
plt.ylabel('BER')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('./figures_BPSK/Tx10Rx10.png')