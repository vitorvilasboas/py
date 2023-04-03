# -*- coding: utf-8 -*-
from numpy import asarray, convolve, dot, ones, zeros
from scipy.linalg import pinv

from sin_basis import make_basis


def filtering(X, fs=250., fl=8., fh=30., m=40):
    # Filtragem usando projeção em base de senos e cossenos.
    # X = tensor de dados, organizado em um tensor de dimensão (ne, nc, nt):
    # ne = Número de épocas (144 no total, 72 para cada classe)
    # nc = Número de canais (22)
    # nt = W_size = Número de amostras no tempo (500, 2 segundos em 250 Hz)
    # fs = Frequencia de amostragem
    # fl = Frequencia mínima

    n_epochs = X.shape[0]
    n_channels = X.shape[1]
    W_Size = X.shape[2]
    # W_Size = X

    # Com a função make_basis criamos uma base de senos e cossenos, de dimensão (nt, 2*m)
    B = make_basis(W_Size, m, fl/fs, fh/fs)


    # G0 é a matriz de projeção, que liga diretamente o sinal original e o sinal projetado na base de interesse
    G0 = dot(pinv(dot(B.T, B)), B.T)

    # Para cada época do vetor X fazemos esta projeção, multiplicando a matriz X[k,:,:] por G0

    XF = asarray([dot(X[k, :, :], G0.T) for k in range(n_epochs)])
    # XF = asarray(dot(X, G0.T)) # for k in range(n_epochs)])


    '''
    SMOOTH = 0
    if SMOOTH:
        v = ones((4)) / 4
        XFC = zeros((n_epochs, n_channels, XF.shape[2] + len(v) - 1))
        for i in range(n_epochs):
            for j in range(n_channels):
                XFC[i,j,:] = convolve(XF[i,j,:], v)
        XF = XFC
    '''

    return XF
