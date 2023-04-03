# -*- coding: utf-8 -*-
import re
import os
import math
import pickle
import itertools
import numpy as np
from time import time
import scipy.signal as sp
from sklearn.svm import SVC
from scipy.stats import norm
from datetime import datetime
from scipy.fftpack import fft
from scipy.linalg import eigh
from sklearn.pipeline import Pipeline
from hyperopt import base, fmin, tpe, hp
from scipy.signal import lfilter, butter
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from scripts.bci_utils import BCI, CSP, extractEpochs, nanCleaner

if __name__ == "__main__":
    class_ids = [1, 2]
    fl, fh, Tmin, Tmax = 8, 30, 2, 3.5
    
    n_bands = 11 # set
    filtro = 'FFT'
    
    dataT, eventsT, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01T.npy', allow_pickle=True)
    dataV, eventsV, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01E.npy', allow_pickle=True)
            
    fs = info['fs']
    smin = math.floor(Tmin * fs)
    smax = math.floor(Tmax * fs) 
    
    epochsT, labelsT = extractEpochs(dataT, eventsT, smin, smax, class_ids)
    epochsV, labelsV = extractEpochs(dataV, eventsV, smin, smax, class_ids)
    XT = [ epochsT[np.where(labelsT==i)] for i in class_ids] # Extrair épocas de cada classe
    XV = [ epochsV[np.where(labelsV==i)] for i in class_ids]
    
    XT = np.concatenate([XT[0],XT[1]]) # Dados de treinamento das classes A, B
    XV = np.concatenate([XV[0],XV[1]]) # Dados de validação das classes A, B
    y = np.concatenate([np.zeros(int(len(XT)/2)), np.ones(int(len(XT)/2))]) # Vetor gabarito

    # Make sub-bands limits and Temporal/Spectral filtering
    buffer = smax - smin
    T = Tmax-Tmin
    nf = fs/2
    
    XTF = []
    if filtro == 'FFT': # Only Spectral filtering
        rf = fs/buffer
        bin0 = int(fl/rf)
        binN = int(fh/rf)
        #print('rf:', nf/fs, fs/buffer, 1/T, fs/len(XT[0,0,:]))
        
        filtered = fft(XT)
        REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1)) 
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        XT_FFT = np.transpose(filtered, (1, 2, 0))       
        
        n_bins = len(XT_FFT[0,0,:]) # ou (fh-fl)*4 # Número total de bins de frequencia 
        step = int(n_bins / n_bands)
        size = int(step / 0.5) # tamanho fixo p/ todas sub bandas. overlap em 50%
        
        for i in range(n_bands):
            bin_ini = i*step
            bin_fim = i*step+size
            if bin_fim >= n_bins: bin_fim = n_bins-1
            XTF.append(XT_FFT[:, :, bin_ini:bin_fim])

    elif filtro == 'IIR':
        n_bins = fh-fl
        step = int(n_bins / n_bands)
        size = int(step / 0.5)
        for i in range(n_bands):
            fl_sb = i*step + fl
            fh_sb = i*step + fl + size
            if fl_sb == 0: fl_sb = 0.001
            if fh_sb > fh: fh_sb = fh
            b, a = butter(5, [fl_sb/nf, fh_sb/nf], btype='bandpass') # to filt IIR
            XTF.append(lfilter(b, a, XT))
            
    
    csp = [CSP(n_components=6) for i in range(n_bands)]
    clf = [LinearDiscriminantAnalysis() for i in range(n_bands)]
    
    for i in range(n_bands): csp[i].fit(XTF[i], y)
    
    XT_CSP = [csp[i].transform(XTF[i]) for i in range(n_bands)]
    
    for i in range(n_bands): clf[i].fit(XT_CSP[i], y)

    # Para cada banda, esse LDA treinado transforma os dados de treinamento e validação.
    # Relembrando, não é feita uma classificação 'dura', o resultado contínuo do LDA é mantido para construir um
    # vetor de características de dimensão (n_bands)
    SCORE_T = np.zeros((len(XT), n_bands)) #(144, n_bands)
    for i in range(n_bands): SCORE_T[:,i] = np.ravel(clf[i].transform(XT_CSP[i])) # classificaçoes de cada época nas N sub bandas - auto validação
        
    # Treino Meta-classificador Bayesiano
    # dados treinamento separados em classe 0 e 1. 
    # calculo média e desvio padrão entre épocas, de forma que os vetores sejam de dimensão n_bands
    SCORE_T0 = SCORE_T[y == 0, :] # separa os scores de cada sub-banda referentes a classe A
    m0 = np.mean(SCORE_T0, axis=0) # media dos scores LDA em cada sub-banda entre todas as épocas da classe A
    std0 = np.std(SCORE_T0, axis=0) # dp dos scores LDA em cada sub-banda entre todas as épocas da classe A
    
    SCORE_T1 = SCORE_T[y == 1, :] # separa os scores de cada sub-banda referentes a classe B
    m1 = np.mean(SCORE_T1, axis=0) # idem - media dos scores LDA para classe B
    std1 = np.std(SCORE_T1, axis=0) # idem - dp dos scores LDA para classe B

    p0 = norm(m0, std0) # projeta uma distribuição normal/gaussiana a partir das médias (m0,m1) e dp (std0,std1)
    p1 = norm(m1, std1) # p0 é uma espécie de filtro bayesiano favorável à classe A e p1 à classe B
    
    SCORE_T_NORMAL_0 = p0.pdf(SCORE_T) # gaussiana centrada na média referente a classe A (m0) e dp (std0)
    SCORE_T_NORMAL_1 = p1.pdf(SCORE_T) # gaussiana centrada na média referente a classe B (m1) e dp (std1)
    
    # scores aplicados nas funções de densidade de probabilidade de cada classe e calculo da razão entre elas.
    # resultado positivo de META_SCORE_T indica que uma época é melhor representada pela função p0,
    # já que p0.pdf(SCORE_T) > p1.pdf(SCORE_T). Caso META_SCORE_T seja negativo a melhor representação é por p1.
    META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
    
    svc = SVC(kernel="linear", C=10**-4, probability=True)
    svc.fit(META_SCORE_T, y)
    
    
    ###### Test ALL set evaluate epochs
    # XVF = []
    # if filtro == 'FFT': # Only Spectral filtering
    #     bin0 = int(fl * 2)
    #     binN = int(fh * 2)

    #     filtered = fft(XV)
    #     REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
    #     IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1))
    #     filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
    #     XV_FFT = np.transpose(filtered, (1, 2, 0))
        
    #     n_bins = len(XT_FFT[0,0,:])
    #     step = int(n_bins / n_bands)
    #     size = int(step / 0.5)
        
    #     for i in range(n_bands):
    #         bin_ini = i*step
    #         bin_fim = i*step+size
    #         if bin_fim >= n_bins: bin_fim = n_bins-1
    #         XVF.append(XV_FFT[:, :, bin_ini:bin_fim])

    # elif filtro == 'IIR':
    #     n_bins = fh-fl
    #     step = int(n_bins / n_bands)
    #     size = int(step / 0.5)
        
    #     for i in range(n_bands):
    #         fl_sb = i*step + fl
    #         fh_sb = i*step + fl + size
    #         if fl_sb == 0: fl_sb = 0.001
    #         if fh_sb > fh: fh_sb = fh
    #         b, a = butter(5, [fl_sb/125, fh_sb/125], btype='bandpass') # to filt IIR
    #         XVF.append(lfilter(b, a, XV))
    
    # SCORE_V = np.zeros((len(XV), n_bands))
    
    # # XV_CSP = [csp[i].transform(XVF[i]) for i in range(n_bands)]
    # # for i in range(n_bands): SCORE_V[:,i] = np.ravel(clf[i].transform(XV_CSP[i]))
    
    # for i in range(n_bands):
    #     XV_CSP = csp[i].transform(XVF[i])
    #     SCORE_V[:, i] = np.ravel(clf[i].transform(XV_CSP)) # validação
    
    # META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
    # scores = svc.predict(META_SCORE_V)
    # acc = np.mean(scores == y) 
    # print ('A1', str(round(acc*100, 2))+'%')
    
    
    
    ###### Test UNIQUE evaluate epoch
    # XVfull = XV
    # for k in range(len(XVfull)):
    #     XV = XVfull[k]
    XV = XV[0]
    XVF = []
    if filtro == 'FFT': # Only Spectral filtering
        filtered = fft(XV)
        # bin0 = int(fl * 2)
        # binN = int(fh * 2)
        bin0 = int(fl / (250/len(filtered[0,:]))) #int(fl * (fs/nf))
        binN = int(fh / (250/len(filtered[0,:]))) #int(fh * (fs/nf))
        REAL = np.transpose(np.real(filtered)[:,bin0:binN])
        IMAG = np.transpose(np.imag(filtered)[:,bin0:binN])
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
        XV_FFT = np.transpose(filtered)
        
        n_bins = len(XT_FFT[0,:])
        step = int(n_bins / n_bands)
        size = int(step / 0.5)
        
        for i in range(n_bands):
            bin_ini = i*step
            bin_fim = i*step+size
            if bin_fim >= n_bins: bin_fim = n_bins-1
            XVF.append(XV_FFT[:, bin_ini:bin_fim])

    elif filtro == 'IIR':
        n_bins = fh-fl
        step = int(n_bins / n_bands)
        size = int(step / 0.5)
        
        for i in range(n_bands):
            fl_sb = i*step + fl
            fh_sb = i*step + fl + size
            if fl_sb == 0: fl_sb = 0.001
            if fh_sb > fh: fh_sb = fh
            b, a = butter(5, [fl_sb/125, fh_sb/125], btype='bandpass') # to filt IIR
            XVF.append(lfilter(b, a, XV))
    
    
    XV_CSP = []
    for i in range(n_bands):
        XT = np.dot(csp[i].filters_, XVF[i])
        XV_CSP.append(np.log(np.mean(XT ** 2, axis=1)))
    
    
    SCORE_V = np.zeros(n_bands)
    for i in range(n_bands): 
        SCORE_V[i] = np.ravel(clf[i].transform(XV_CSP[i].reshape(1,-1)))
    

    META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
    score = svc.predict(META_SCORE_V.reshape(1,-1))
    
    proba = svc.predict_proba(META_SCORE_V.reshape(1,-1))
    
    print(score, proba)
        
    #if score != y[k]: print('ERRO', k)