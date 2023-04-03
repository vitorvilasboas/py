# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from numpy import arange, asarray, cos, cov, dot, linspace, pi, sin, zeros
from numpy.linalg import lstsq, norm, pinv
from numpy.random import normal

q = 200
Fs = 0.5

X = normal(0, 1, q)
n = arange(q) + 0.5

def project(X0, X):
    COV = dot(X0, X0.T)
    G0 = dot(X0.T, pinv(COV))
    P0 = dot(G0, X0)
    Xhat = dot(P0, X)
    error = norm(X-Xhat) / norm(X)
    return error

error_cos = zeros(q/2)
error_sin = zeros(q/2)
error_both = zeros(q/2)

for m in arange(1, q/2):

    X0_cos = asarray([cos(2*pi*f*n) for f in linspace(0, Fs, 2*m)])
    error_cos[m-1] = project(X0_cos, X)

    X0_sin = asarray([sin(2*pi*f*n) for f in linspace(0, Fs, 2*m)])
    error_sin[m-1] = project(X0_sin, X)
    
    X0_both = asarray([cos(2*pi*f*n) for f in linspace(0, Fs, m)]+
                      [sin(2*pi*f*n) for f in linspace(0, Fs, m)])
    error_both[m-1] = project(X0_both, X)

plt.figure(figsize=(7,4))
plt.plot(error_cos, label='Cossenos')
plt.plot(error_sin, label='Senos')
plt.plot(error_both, label='Senos + Cossenos')
plt.legend(loc=3)
plt.xlabel(u'Número de vetores na base $X_0$')
plt.ylabel(u'Erro de reconstrução')
plt.show()
#plt.savefig('reconst.pdf', bbox_inches='tight')
