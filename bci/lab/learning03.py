# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg as lg
from scipy.signal import lfilter, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

######### CALIBRAÇÃO #########
b, a = butter(5, [8/125, 30/125], btype='bandpass')
Z = np.load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01T_1.npy', 'rb'))
Z = Z[:,:,125:625]
Z0 = lfilter(b, a, Z)
Z = np.load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01T_2.npy', 'rb'))
Z = Z[:,:,125:625]
Z1 = lfilter(b, a, Z)

S0 = np.zeros((22, 22))
S1 = np.zeros((22, 22))
for e in range(72):
    S0 += np.dot(Z0[e], Z0[e].T)
    S1 += np.dot(Z1[e], Z1[e].T)
[D, W] = lg.eigh(S0, S0 + S1)
ind = np.empty(22, dtype=int)
ind[::2] = np.arange(22 - 1, 22 // 2 - 1, -1) 
ind[1::2] = np.arange(0, 22 // 2)
W = W[:, ind]
W = W.T[:6]

X = np.concatenate([Z0,Z1])
X = np.asarray([np.dot(W, e) for e in X])
X1 = np.log(np.mean(X ** 2, axis=2))

clf = LinearDiscriminantAnalysis()
y = np.concatenate([np.zeros(72), np.ones(72)])
clf.fit(X1, y)

######### OPERAÇÃO 1 epoca #########
EEG = np.load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01E_2.npy', 'rb'))
zv = EEG[0,:,125:625]
zv = lfilter(b, a, zv)
xv = np.dot(W, zv)
xv1 = np.log(np.mean(xv ** 2, axis=1))
xv2 = xv1.reshape(1, -1)
saida_lda = clf.predict(xv2)
print(saida_lda)
# Expected 2D array, got 1D array instead. Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.