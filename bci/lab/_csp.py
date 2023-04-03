#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:16:23 2020
@author: vboas
"""
import math
import cmath
import itertools
import numpy as np
from scipy.linalg import eigh
from scipy.fftpack import fft
from scripts.bci_utils import extractEpochs
from sklearn.metrics import cohen_kappa_score
from scipy.signal import lfilter, butter, iirfilter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

###################
# import numpy as np
# idx = np.arange(0, 144, 1, dtype=int)
# idx_teste = np.random.choice(144, size=44, replace=False)
# idx = np.asarray(list(set(idx).difference(idx_teste)))
# idx_treino = np.random.choice(idx, size=50, replace=False)
# idx_valida = np.asarray(list(set(idx).difference(idx_treino)))
# idx_teste.sort()
# idx_treino.sort()
# idx_valida.sort()
####################


dataT, eventsT, infoT = np.load('/mnt/dados/eeg_data/IV2a/npy/A01T.npy', allow_pickle=True)
dataV, eventsV, infoV = np.load('/mnt/dados/eeg_data/IV2a/npy/A01E.npy', allow_pickle=True)

Fs = infoT['fs']
class_ids = [1,2]
smin = math.floor(0.5 * Fs)
smax = math.floor(2.5 * Fs)
epochsT, labelsT = extractEpochs(dataT, eventsT, smin, smax, class_ids)
epochsV, labelsV = extractEpochs(dataV, eventsV, smin, smax, class_ids)

ZT = [epochsT[np.where(labelsT==i)] for i in class_ids]
ZV = [epochsV[np.where(labelsV==i)] for i in class_ids]

ZT = np.r_[ZT[0],ZT[1]]
ZV = np.r_[ZV[0],ZV[1]]

tT = np.r_[class_ids[0]*np.ones(int(len(ZT)/2)), class_ids[1]*np.ones(int(len(ZT)/2))]
tV = np.r_[class_ids[0]*np.ones(int(len(ZV)/2)), class_ids[1]*np.ones(int(len(ZV)/2))]

fl, fh = 0, 4

buffer_len = smax - smin
dft_res_freq = Fs/buffer_len # resolução em frequência fft
dft_size_band = round(2/dft_res_freq) # 2 representa sen e cos que foram separados do componente complexo da fft intercalados
bmin = fl * dft_size_band
bmax = fh * dft_size_band

fft_out = fft(ZT)
REAL = np.transpose(np.real(fft_out), (2, 0, 1))
IMAG = np.transpose(np.imag(fft_out), (2, 0, 1))
fft_out = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
XT_FFT = np.transpose(fft_out, (1, 2, 0))
# XT_FFT = abs(fft_out)

fft_out = fft(ZV)
REAL = np.transpose(np.real(fft_out), (2, 0, 1))
IMAG = np.transpose(np.imag(fft_out), (2, 0, 1))
fft_out = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
XV_FFT = np.transpose(fft_out, (1, 2, 0))
# XV_FFT = abs(fft_out)

XT = XT_FFT[:, :, bmin:bmax]
XV = XV_FFT[:, :, bmin:bmax]

# nyq = 0.5 * Fs
# low = fl / nyq
# high = fh / nyq
# if low == 0: low = 0.001
# if high >= 1: high = 0.99
# b, a = butter(5, [low, high], btype='bandpass')
# # b, a = iirfilter(5, [low,high], btype='band')
# # XT = lfilter(b, a, ZT) 
# XT = filtfilt(b, a, ZT)

ncomp = 6
e, c, s = XT.shape
classes = np.unique(tT)   
Xa = XT[classes[0] == tT,:,:]
Xb = XT[classes[1] == tT,:,:]

Sa = np.zeros((c, c)) 
Sb = np.zeros((c, c))
for i in range(int(e/2)):
    # Sa += np.dot(Xa[i], Xa[i].T) / Xa[i].shape[-1] # sum((Xa * Xa.T)/q)
    # Sb += np.dot(Xb[i], Xb[i].T) / Xb[i].shape[-1] # sum((Xb * Xb.T)/q)
    Sa += (Xa[i] @ Xa[i].T) / np.trace(Xa[i] @ Xa[i].T)
    Sb += (Xb[i] @ Xb[i].T) / np.trace(Xb[i] @ Xb[i].T) # soma das matrizes de covariancias de todas as época da classe
Sa /= (len(Xa))
Sb /= (len(Xb))

[D, W] = eigh(Sa, Sa + Sb)
ind = np.empty(c, dtype=int)
ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
ind[1::2] = np.arange(0, c // 2)
W = W[:, ind]
Wf = W.T[:ncomp]
       
YT = np.asarray([np.dot(Wf, ep) for ep in XT]) # aplicação filtro em todas as épocas de treino... e test (W.T * XTi)
YV = np.asarray([np.dot(Wf, ep) for ep in XV])

XT_CSP = np.log(np.mean(YT ** 2, axis=2)) #log da energia média do sinal (x^2)/q em cada "novo sinal" gerado pela aplicação dos filtros CSP
XV_CSP = np.log(np.mean(YV ** 2, axis=2))

XT_CSP1 = np.log(np.var(YT, axis=2))
XV_CSP1 = np.log(np.var(YV, axis=2))

clf = LDA()
clf.fit(XT_CSP1, tT)
scores_labels = clf.predict(XV_CSP1)
acc = np.mean(scores_labels == tV)
kappa = cohen_kappa_score(scores_labels, tV)
print('Accuracy:', round(acc,4))
print('kappa:', round(kappa,4))

# XV_CSPi = np.log(np.mean(YV[0] ** 2, axis=1))
# teste = [ np.var(YT[i], axis=2) for i in range(YT.shape[0])]

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fl, fh = 0, 4
Z = ZT[0] # epoca 0
X = fft(Z)

###################

bmin = fl * round(1 / (Fs / (smax - smin) ) )
bmax = fh * round(1 / (Fs / (smax - smin) ) )
XX = X[:, bmin:bmax]
C = XX @ XX.T

# XX = XX[0]
C1 = np.zeros((22,22))
for i in range(XX.shape[-1]):
    XX1 = XX[:,i].reshape(-1,1)
    C1 += XX1 @ XX1.T

###################

bmin = fl * round(2 / (Fs / (smax - smin) ) )
bmax = fh * round(2 / (Fs / (smax - smin) ) )
X0 = np.transpose(list(itertools.chain.from_iterable(zip(np.imag(X).T, np.real(X).T))))
X1 = X0[:, bmin:bmax]
C1 = (X1 @ X1.T)/X1.shape[-1]

C2 = np.zeros((22,22))
for i in range(0, X1.shape[-1], 2):
    X2 = X1[:,i:i+2]
    C2 += (X2 @ X2.T)
C2 = C2/X1.shape[-1]

### Conclusão: C1 == C2 (Covariancia entre os varios bins == soma das var de cada bin)

Cov = C1 / np.trace(C1)

Cov2 = np.diag(X1 @ X1.T) / np.trace(X1 @ X1.T) 

dim = np.ndim(C1) # retorna a primeira dimensão da matriz ou dimensão do vetor
trace = np.trace(C1) # sum(diag)
diag = np.diag(C1)
mean = np.mean(diag)
f = np.log(mean)

det = np.linalg.det(C1) # determinante de C

inv = C1.dot(np.linalg.inv(C1)) # inversa de C
inv = np.dot(C1, np.linalg.inv(C1))

# # x = np.random.rand(4,4)
# # np.eye(4)
# # np.diag(x @ x.T) / np.trace(x @ x.T)
