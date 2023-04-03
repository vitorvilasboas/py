# -*- coding: utf-8 -*-
from numpy import concatenate, imag, real, zeros, arange, abs, linspace, dtype, fromfile, transpose, ones
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from general.filtering_proj import filtering as filtPROJ
from general.filtering_fft_v1 import filtering as filtFFT


def windowing(X, t_0, t_start, t_end, fs):
    '''
    X[0][0] = A0sT_a(72,22,1000)
    X[0][1] = A0sE_a
    X[1][0] = A0sT_b
    X[1][1] = A0sT_b
    '''
    atraso = 1
    W_Start = int((t_start - t_0) * fs)
    W_End = int((t_end - t_0) * fs)
    # Xa = [[],[]]
    for i in range(2):
        for j in range(2):
            # X[i][j] = X[i][j][:, :, W_Start:W_End]
            Xa = X[i][j][:, :, W_Start:W_End]
            for cont in range(1, atraso + 1):
                Xb = X[i][j][:, :, W_Start + cont:W_End + cont]
                Xa = transpose(Xa, (1,0,2))
                Xb = transpose(Xb, (1,0,2))
                Xa = concatenate([Xa, Xb])
                Xa = transpose(Xa, (1, 0, 2))
                #print(Xa.shape)
            X[j].append(Xa)
    return X


folder = "D:/bci_tools/dset42a/fdt/epocas/"
subject = 9
classes = [1, 2]
X = [[], []]
set = ['T_', 'E_']
for i in range(2):
    for j in range(2):
        path = folder + '/A0' + str(subject) + set[j] + str(classes[i]) + '.fdt'
        fid = open(path, 'rb')
        data = fromfile(fid, dtype('f'))
        data = data.reshape((72, 1000, 22))
        X[j].append(transpose(data, (0,2,1)))

X = windowing(X, 2, 2.5, 4.5, 250)
# print(X[0][0].shape)






'''
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
'''