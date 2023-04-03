# -*- coding: utf-8 -*-
from numpy import concatenate, imag, real, zeros
# from numpy.fft import fft, rfft2, fft2, rfft
from scipy.fftpack import fft, ifft


def filtering(X, fl=0., fh=51.):
    fs = 250
    n_epochs = X.shape[0]
    n_channels = X.shape[1]
    W_Size = X.shape[2]
    FL = int(W_Size * fl / fs)
    FH = int(W_Size * fh / fs)
    FN = FH - FL
    XF = zeros((n_epochs, n_channels, 102))  # FFT simples = FN; FFT concatenada ou INTERCALADA = 2*FN;
    for k in range(n_epochs):
        for l in range(n_channels):
            XFFT = fft(X[k, l, :])
            # FFT simples: considera valores complexos - ajustar XF(ne, nc, FN)
            # XF[k, l, :] = XFFT[:FN]  # :FN ou :200

            # FFT Concatenando partes (real)(imag) do vetor de complexo resultante
            XREAL = real(XFFT)[FL:102]  # :102 representando as 51 primeiras frquências absolutas
            XIMAG = imag(XFFT)[FL:102]
            XF[k, l, :] = concatenate([XREAL, XIMAG])

            # FFT INTERCALANDO partes (real)(imag) do vetor de complexo resultante
            # XREAL = real(XFFT)[FL:FH]  # :102 representando as 51 primeiras frquências absolutas
            # XIMAG = imag(XFFT)[FL:FH]
            # j = 0
            # for i in range(0, 204, 2):  # intercala os coeficientes das partes real e imaginária de cada frequência
            #    XF[k, l, i] = XREAL[j]
            #    XF[k, l, i + 1] = XIMAG[j]
            #    j += 1

    return XF