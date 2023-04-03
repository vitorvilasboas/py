from numpy import concatenate, imag, real, zeros, arange, abs, linspace, dtype, fromfile, transpose
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from general.filtering_proj import filtering as filtPROJ
from general.filtering_fft_v1 import filtering as filtFFT

# LOAD DATASET
path = 'D:/bci_tools/dset42a/fdt/epocas/A03T_2.fdt'
fid = open(path, 'rb')
data = fromfile(fid, dtype('f'))
data = data.reshape((72, 1000, 22))
X = transpose(data, (0, 2, 1))
XT = X[:, :, 0:500]
fs = 250

# PLOT FFT
XTF = zeros((144, 22, 200))
FFT = filtFFT(XT, fl=0, fh=51)
XTF[1, 1, :] = FFT[1, 1, :]
fig, ax = plt.subplots()
ax.plot((XTF[1, 1, :]))

# PLOT Projeção
XTF = zeros((144, 22, 200))
FFT = filtPROJ(XT, fl=0, fh=51, m=100)
XTF[1, 1, :] = FFT[1, 1, :]
fig, ax = plt.subplots()
ax.plot((XTF[1, 1, :]))

plt.show()
