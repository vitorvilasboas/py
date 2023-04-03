# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import lfilter, butter, welch
from scipy.fftpack import fft

A01T_RH = np.load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01T_1.npy', 'rb'))
Z = np.array(A01T_RH[0,:,125:625])
Z = Z.T

b, a = butter(5, [8/125, 30/125], btype='bandpass')
Z = lfilter(b, a, Z)

z0 = Z[:,0] # canal 0
z1 = Z[:,1] # canal 1
z14 = Z[:,14] # canal 14
z18 = Z[:,18] # canal 18

fig, ax = plt.subplots()
#ax = plt.plot(z18**2)
ax = plt.plot(z18)
ax = plt.plot(np.abs(z18))


vz0 = np.var(z0)
vz1 = np.var(z1)
vz14 = np.var(z14)
vz18 = np.var(z18)


np.corrcoef(Z[:,14], Z[:,18])

plt.scatter(Z[:,14], Z[:,18])

sns.distplot(Z[:,18], hist = False, kde = True,
             bins = 6, color = 'blue',
             hist_kws={'edgecolor': 'black'})





f, X = welch(Z[:,18], 250)
f, Xa = welch(Z[:,14], 250)
fig, ax = plt.subplots()
ax = plt.plot(f, X)
ax = plt.plot(f, Xa)


f, X = welch(Z[:,18], 250)

Xaa = fft(Z[:,18])
plt.plot(np.abs(Xaa[:60]))