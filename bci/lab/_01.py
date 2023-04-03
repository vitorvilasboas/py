#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:00:49 2020
@author: vboas
"""
import math
import itertools
import numpy as np
from time import time
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.linalg import eigh
from scipy.fftpack import fft
from sklearn.metrics import cohen_kappa_score
from bci_utils import labeling, extractEpochs
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import lfilter, butter, iirfilter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression


data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01.npy', allow_pickle=True)
Fs = info['fs']
class_ids = [1,2]
# tmin, tmax = 0.5, 2.5
tmin = [0, 0.5, 1, 1.5, 2]
tmax = [2, 2.5, 3, 3.5, 4]

ST,SV = [], []
for k in range(len(tmin)):

    smin, smax = math.floor(Fs*tmin[k]), math.floor(Fs*tmax[k])
    
    epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
    epochs = [ epochs[np.where(labels == i)] for i in class_ids ]
    Za, Zb = epochs[0], epochs[1]
    
    ZT = np.r_[ Za[:int(len(Za)/2)], Zb[:int(len(Zb)/2)]  ]
    ZV = np.r_[ Za[int(len(Za)/2):], Zb[int(len(Zb)/2):]  ]
    
    tT = np.r_[np.ones(len(ZT)//2)*class_ids[0], np.ones(len(ZT)//2)*class_ids[1]].astype(int)
    tV = np.r_[np.ones(len(ZV)//2)*class_ids[0], np.ones(len(ZV)//2)*class_ids[1]].astype(int)
    
    nbands = 9
    f_low, f_high = 0, 40
    
    n_samples = smax - smin
    n_bins = f_high - f_low
    overlap = 0.5 
    step = n_bins / (nbands+1)
    size = step / overlap
    
    nyq = Fs/2
    
    XT,XV, sub_bands = [], [],[]
    for i in range(nbands):
        fl_sb = i * step + f_low
        fh_sb = i * step + size + f_low
        sub_bands.append([fl_sb, fh_sb])
            
        low, high = fl_sb/nyq, fh_sb/nyq
        if low == 0: low = 0.001
        if high >= 1: high = 0.999
        b, a = butter(5, [low,high], btype='bandpass')
        
        XT.append(lfilter(b, a, ZT)) 
        XV.append(lfilter(b, a, ZV))
     
    # ini, fim = f_low, size
    # for i in range(5):print([ini,fim]); ini+=step; fim+=step
    
    ncsp = 8
    filters_ = []
    for i in range(nbands):
        e, c, s = XT[i].shape  
        Xa = XT[i][class_ids[0] == tT,:,:]
        Xb = XT[i][class_ids[1] == tT,:,:]
        
        x = Xa[0]
        Ca = x @ x.T
        
        Sa, Sb = np.zeros((c, c)), np.zeros((c, c))
        for i in range(int(e/2)):
            Sa += np.dot(Xa[i,:,:], Xa[i,:,:].T) / Xa[i].shape[-1] # sum((Xa * Xa.T)/q)
            Sb += np.dot(Xb[i,:,:], Xb[i,:,:].T) / Xb[i].shape[-1] # sum((Xb * Xb.T)/q)
        Sa /= len(Xa)
        Sb /= len(Xb)
        
        [D, W] = eigh(Sa, Sa + Sb)
        ind = np.empty(c, dtype=int)
        ind[0::2] = np.arange(c-1, c//2-1, -1) 
        ind[1::2] = np.arange(0, c//2)
        W1 = W[:, ind]
        filters_.append(W1.T[:ncsp])
    
    
    Y = [np.asarray([np.dot(filters_[i], ep) for ep in XT[i]]) for i in range(nbands)] # Spatial Filtering
    
    XTCSP = [np.log(np.mean(Y[i]**2, axis=2)) for i in range(nbands)] # Feature extraction
    # um escalar que representa a energia média do sinal nos ncsp "novos canais" em cada época EM CADA SUB-BANDA
    # XTCSP1 = [np.log(np.var(Y[i], axis=2)) for i in range(nbands)] # Feature extraction
    
    Y = [ np.asarray([ np.dot(filters_[i], ep) for ep in XV[i] ]) for i in range(nbands) ] # Spatial Filtering
    XVCSP = [ np.log(np.mean(Y[i]**2, axis=2)) for i in range(nbands) ] # Feature extraction

    lda = [LDA() for i in range(nbands)]
    for i in range(nbands): lda[i].fit(XTCSP[i], tT)
    
    SCORE_T = np.asarray([np.ravel(lda[i].transform(XTCSP[i])) for i in range(nbands)]).T
    SCORE_V = np.asarray([np.ravel(lda[i].transform(XVCSP[i])) for i in range(nbands)]).T # Score LDA 
    
    SCORE_T2 = np.asarray([np.ravel(lda[i].predict(XTCSP[i])) for i in range(nbands)]).T
    SCORE_V2 = np.asarray([np.ravel(lda[i].predict(XVCSP[i])) for i in range(nbands)]).T

    # print(SCORE_T[0], SCORE_V[0])
    
    ST.append(SCORE_T)
    SV.append(SCORE_V)

ST = np.transpose(ST, (2,1,0))
SV = np.transpose(SV, (2,1,0))

SCORE_T = np.mean(ST, axis=2).T
SCORE_V = np.mean(SV, axis=2).T

lr = [LinearRegression() for i in range(nbands)]
for i in range(nbands): lr[i].fit(ST[i], tT)
SCORE_T = np.asarray([np.ravel(lr[i].predict(ST[i])) for i in range(nbands)]).T
SCORE_V = np.asarray([np.ravel(lr[i].predict(SV[i])) for i in range(nbands)]).T

SCORE_TA = SCORE_T[tT == class_ids[0], :]
SCORE_TB = SCORE_T[tT == class_ids[1], :]

mA = np.mean(SCORE_TA, axis=0)
dpA = np.std(SCORE_TA, axis=0)

mB = np.mean(SCORE_TB, axis=0)
dpB = np.std(SCORE_TB, axis=0)

NA = norm(mA, dpA) # projeta uma distribuição normal/gaussiana a partir das médias (m0,m1) e dp (std0,std1) dos scores LDA em cada sub-banda ... 
NB = norm(mB, dpB) # entre todas as épocas de cada classe. p0 é uma espécie de filtro bayesiano favorável à classe A e p1 à classe B
META_SCORE_T = np.log(NA.pdf(SCORE_T) / NB.pdf(SCORE_T))

# SCORE_TNA = NA.pdf(SCORE_T)
# SCORE_TNB = NB.pdf(SCORE_T)
# SCORE_TDIV = np.log(SCORE_TNA/SCORE_TNB)

clf_final = SVC(kernel='linear', C=1e-4, probability=True)
clf_final.fit(META_SCORE_T, tT)

META_SCORE_V = np.log(NA.pdf(SCORE_V) / NB.pdf(SCORE_V)) # Bayesian Meta-score

y = META_SCORE_V @ clf_final.coef_.T # y<0=classe A(1) ; y>0=classe B(2)
y_labels = clf_final.predict(META_SCORE_V) # Final classification
y_proba = clf_final.predict_proba(META_SCORE_V)

acc = clf_final.score(META_SCORE_V, tV)
acc = np.mean(y_labels == tV)
kappa = cohen_kappa_score(y_labels, tV)
print('Accuracy:', round(acc,4))
print('kappa:', round(kappa,4))

