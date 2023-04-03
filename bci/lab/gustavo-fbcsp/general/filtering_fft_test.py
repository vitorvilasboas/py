# -*- coding: utf-8 -*-
from numpy import concatenate, imag, real, zeros, where
# from numpy.fft import fft, rfft2, fft2, rfft
from scipy.fftpack import fft, fftfreq


def filtering(X, fl=0., fh=51.):
    fs = 250
    n_epochs = X.shape[0]
    n_channels = X.shape[1]
    W_Size = X.shape[2]
    XF = zeros((n_epochs, n_channels, 200))  # 2 * FN 200 posso alterar utilizar W_Size para n_pontos = n_amostras
    for k in range(n_epochs):
        for l in range(n_channels):
            sig = X[k, l, :]
            sig_fft = fft(sig)
            n = sig.size
            timestep = 0.004
            freq = fftfreq(n, d=timestep)
            freq1 = where(freq > 0)
            power = abs(sig_fft)[freq]
            print(power)
            XF[k, l, :] = power
            '''
            #freq1 = where(freq > 0)
            #print(freq)
            #freq2 = where(freq1 < 52)
            # print(abs(sig_fft)[freq].size)
            #power = abs(sig_fft)[freq1]
            # XF[k, l, :] = power

            sample_freq = fftfreq(sig.size, d=0.004)
            sig_fft = fft(sig)
            pidxs = where(0 < sample_freq)
            freqs = sample_freq[pidxs]
            power = abs(sig_fft)[pidxs]
            XF[k, l, :] = power[0:200]

            #XF[k, l, :] = fft(X[k, l, :])

            # XFFT = fft(X[k, l, :])
            # XREAL = real(XFFT)[FL:FH]  # 100 parte real das 100 primeiras frequências
            # XIMAG = imag(XFFT)[FL:FH]  # 100 parte imaginária das 100 primeiras frequências
            # XF[k, l, :] = concatenate([XREAL, XIMAG])

            # j = 0
            # for i in range(0, 200, 2):  # intercala os coeficientes das partes real e imaginária de cada frequência
            #    XF[k, l, i] = XREAL[j]
            #    XF[k, l, i + 1] = XIMAG[j]
            #    j += 1
'''

    return XF

    # XFFT = rfft2(X[k, :, :])  # Opção de FFT de duas dimensões. Nesse caso incluíria o indice dos canais
