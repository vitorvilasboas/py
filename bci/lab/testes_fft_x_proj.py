# -*- coding: utf-8 -*-
from __future__ import division

from numpy import dtype, fromfile, transpose, real, imag
from numpy import arange, cos, pi, sin, zeros, asarray, dot, abs
from numpy.fft import fftfreq
from scipy.linalg import pinv
from scipy.fftpack import fft
import matplotlib.pyplot as plt

path = '/mnt/dados/bci_tools/dset42a/fdt/epocas_t2/A03T_2.fdt'
fid = open(path, 'rb')
data = fromfile(fid, dtype('f'))
data = data.reshape((72, 1000, 22))
X = transpose(data, (0, 2, 1))
XI = X[1, 1, 0:500]
# XI = arange(0,500)

fs = 250
fl = 0
fh = 51
m = 102
fi = fl/fs
fm = fh/fs
t = arange(len(XI))
B = zeros((len(XI), 2 * m))
for i in range(m):
    f = fi + i / m * (fm - fi)
    b_sin = sin(2 * pi * f * t)
    b_cos = cos(2 * pi * f * t)
    # B[:, 2 * i] = b_sin
    # B[:, 2 * i + 1] = b_cos
    B[:,i] = b_cos
    B[:,m+i] = b_sin

G0 = dot(pinv(dot(B.T, B)), B.T)
#XF = asarray([dot(XT, G0.T)])
XP = dot(XI, G0.T)*250
#XP = dot(XI,B)
print(len(XP),len(X[0]))
XF = fft(XI)
freq = fftfreq(XI.size, 0.5)

XREAL = real(XF)[fl:102]
XIMAG = imag(XF)[fl:102]#[fl:100]
XFFT = zeros((200))
j = 0
for i in range(0, 200, 2):
   XFFT[i] = XIMAG[j]
   XFFT[i + 1] = XREAL[j]
   j += 1

print(XFFT[:10])

#fig, ax = plt.subplots()
# ax.plot(freq, XFFT)
fig, ax = plt.subplots()
ax.plot(XP[m:])
ax.plot(-XIMAG)
plt.show()

fig, ax = plt.subplots(X)

"""
XI = [arange(0,6)]
XT = dot(transpose(XI), XI)
print(XT)
w = exp(-i)
"""

