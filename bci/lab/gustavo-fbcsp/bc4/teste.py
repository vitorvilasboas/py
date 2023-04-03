# -*- coding: utf-8 -*-
from __future__ import division
from numpy import dot, concatenate, imag, real, zeros, arange, abs, linspace, dtype, fromfile, transpose
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from sys import path
from time import time

path.append('../general')

from filtering_proj import filtering as filtPROJ
from filtering_fft import filtering as filtFFT

# LOAD DATASET
path = 'C:/Users/Josi/OneDrive/datasets/c4_2a/epocas/A03T_2.fdt'
fid = open(path, 'rb')
data = fromfile(fid, dtype('f'))
data = data.reshape((72, 1000, 22))
X = transpose(data, (0, 2, 1))
XT = X[:, :, 0:500]
fs = 250
# N = 6
# ind = arange(0,N)
# B = ind.T*ind
# print(B)

# PLOT FFT
XTF = zeros((144, 22, 500))
XTF = fft(XT[1, 1, :])/500
fig, ax = plt.subplots()
ax.plot(XTF[:200])

# PLOT Projeção
FFT1 = filtPROJ(XT, fl=0, fh=51, m=100)
fig, ax = plt.subplots()
ax.plot((FFT1[1, 1, :]))
plt.show()

