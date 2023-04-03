# -*- coding: utf-8 -*-
# Gera matriz com base senoidal/cossenoidal

from __future__ import division
from numpy import arange, cos, pi, sin, zeros



def make_basis(q, m, fi, fm):
    # Construção da base de senos e cossenos
    # q = Tamanho dos sinais na base (nt)
    # m = Número de senos e cossenos na base
    # fi = Frequencia inicial reduzida (fl/fs)
    # fm = Frequencia máxima reduzida (fh/fs)

    t = arange(q)
    X0 = zeros((q, 2*m))
    for i in range(m):
        # Calculamos a frequencia reduzida correspondente ao bin i
        f = fi + i / m * (fm-fi)

        # Obtemos o seno e o cosseno dessa frequencia
        b_sin = sin(2*pi*f*t)
        b_cos = cos(2*pi*f*t)

        # Colocamos os dois nas posições 2*i e 2*i+1 da base, de forma que ela fique organizada da seguinte forma
        # [Seno_0, Cosseno_0, Seno_1, Cosseno_1, etc]
        X0[:, 2*i] = b_sin
        X0[:, 2*i+1] = b_cos

    return X0

"""
if __name__ == "__main__":

    t_0 = 2.5
    t_start = 2.5
    t_end = 4.5
    fs = 250

    W_Start = (t_start - t_0) * fs
    W_End = (t_end - t_0) * fs
    W_Size = W_End - W_Start
        
    fl = 8.
    fh = 30.
    m = 40
    
    X0 = make_basis(W_Size, m, fl/fs, fh/fs)
    
    # print(X0.shape)
    
    # print('Número de vetores na base: ' + str(2*m))
    # print('Tamanho dos vetores: ' + str(int(W_Size)))
"""

