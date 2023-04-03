# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from numpy import arange, dot, linspace
from numpy.fft import fft
from numpy.linalg import pinv
from numpy.random import normal

def project(X0, X):
    COV = dot(X0, X0.T)
    G0 = dot(X0.T, pinv(COV))
    P0 = dot(G0, X0)
    Xhat = dot(P0, X)
    return Xhat

signal_length = 200
freq_limit = 0.2

# Synthetize random signal
X = normal(0, 1, signal_length)

# For making the basis
n = arange(signal_length) + 0.5

# Number of vectors in the basis
mv = [20, 40, 45]

# Plotting
fsize = 12
fig = plt.figure(figsize(fsize,fsize/3.))

# Loop through the 3 basis
for m, i in zip(mv, arange(1,4)):

    B = freq_limit/m

    # Make dictionary and project the signal into it
    # Not using make_basis: here there are only cosines
    X0 = asarray([cos(2*pi*f*n) for f in arange(0, freq_limit, B)])
    Xhat = project(X0, X)
    
    # Transform signal to frequency domain with FFT
    XHAT = log(abs(fft(Xhat)[0:signal_length/2]))

    # Plot
    plt.subplot(1,3,i)
    plt.plot(linspace(0,0.5,len(XHAT)), XHAT)

    if i == 1:
        plt.ylabel(u'Potência')
    if i == 2:
        plt.xlabel(u'Frequência reduzida')

fig.tight_layout()
plt.show()
#plt.savefig('../relatorio-1/spectrum-m.pdf', bbox_inches='tight')
