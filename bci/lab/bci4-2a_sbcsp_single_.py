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


class CSP():
    def __init__(self, n_components):
        self.n_components = n_components
        # self.filters_ = None
    def fit(self, X, y):
        e, c, s = X.shape
        classes = np.unique(y)   
        Xa = X[classes[0] == y,:,:]
        Xb = X[classes[1] == y,:,:]
        S0 = np.zeros((c, c)) 
        S1 = np.zeros((c, c))
        for epoca in range(int(e/2)):
            # S0 = np.add(S0, np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T), out=S0, casting="unsafe")
            # S1 = np.add(S1, np.dot(Xb[epoca,:,:], Xb[epoca,:,:].T), out=S1, casting="unsafe")
            S0 += np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T) #covA Xa[epoca]
            S1 += np.dot(Xb[epoca,:,:], Xb[epoca,:,:].T) #covB Xb[epoca]
        [D, W] = eigh(S0, S0 + S1)
        ind = np.empty(c, dtype=int)
        ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
        ind[1::2] = np.arange(0, c // 2)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        return self # instruction add because cross-validation pipeline
    def transform(self, X):        
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        XVAR = np.log(np.mean(XT ** 2, axis=2)) # Xcsp
        return XVAR
    
def nanCleaner(epoch):
    """Removes NaN from data by interpolation
    data_in : input data - np matrix channels x samples
    data_out : clean dataset with no NaN samples"""
    for i in range(epoch.shape[0]):
        bad_idx = np.isnan(epoch[i, :])
        epoch[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoch[i, ~bad_idx])
    return epoch

def extractEpochs(data, events, smin, smax, classes):
    cond = False # cond index=True if timestamps[:,1] in [1,2,3,4]
    for cl in classes: cond += (events[:, 1] == cl) 
    idx = np.where(cond)[0] # storing 288 class labels indexis
    t0_stamps = events[idx, 0] # get sample_stamp(posições) relacionadas ao inicio das 288 tentativas de epocas das classes
    sBegin = t0_stamps + smin # vetor que marca as amostras que iniciam cada época
    sEnd = t0_stamps + smax # vetor que contém as amostras que finalizam cada epoca
    n_epochs = len(sBegin)
    n_channels = data.shape[0]
    n_samples = smax - smin
    epochs = np.zeros([n_epochs, n_channels, n_samples])
    labels = events[idx,1] # vetor que contém os indices das 288 épocas das 4 classes
    bad_epoch_list = []
    for i in range(n_epochs): 
        epoch = data[:, sBegin[i]:sEnd[i]]
        if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch # Check if epoch is complete
        else:
            print('Incomplete epoch detected...')
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    return epochs, labels


if __name__ == "__main__":
    
    subjects = range(1,10)
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    eeg_channels = 22
    
    suj = 9 # set 0 a 9
    cl = [1, 2] # set 0 a 5
    n_comp = 6 # set 2 a n_channels
    fs = 250
    fl = 8 
    fh = 30
    nf = fs/2.
    n_bands = 11 # set
    order = 5
    Clog = -4
    smin = int(2.5 * fs)
    smax = int(4.5 * fs)
    overlap = 0.5
    classifier = 'SVM'
    filtro = 'IIR'
    
    path = '/mnt/dados/eeg_data/BCI4_2a/'
    for ds in ['T','E']:
        events = np.load(path + 'npy_sep/A0' + str(suj) + ds + '_events.npy')
        data = np.load(path + 'npy_sep/A0' + str(suj) + ds + '_data.npy')
        data = data[:eeg_channels] # get only 22 EEG channels
        epochs, labels = extractEpochs(data, events, smin, smax, cl)
        #epochs = nanCleaner(epochs) # Correção de NaN nas épocas
        if ds=='T': XT0 = [ epochs[np.where(labels==i)] for i in cl] # Extrair épocas de cada classe
        if ds=='E': XV0 = [ epochs[np.where(labels==i)] for i in cl]
        
    XT = np.concatenate([XT0[0],XT0[1]]) # Dados de treinamento das classes A, B
    XV = np.concatenate([XV0[0],XV0[1]]) # Dados de validação das classes A, B
    y = np.concatenate([np.zeros(int(len(XT)/2)), np.ones(int(len(XT)/2))]) # Vetor gabarito
    
    start = time() # start timer
    
    if filtro == 'FFT': # Spectral filtering
        bin0 = int(fl * (fs/nf))
        binN = int(fh * (fs/nf))
        filtered = fft(XT)
        REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1)) 
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        XT_FFT = np.transpose(filtered, (1, 2, 0))       
        filtered = fft(XV)
        REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1))
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
        XV_FFT = np.transpose(filtered, (1, 2, 0))
       
    # Divide sub-bands
    if filtro == 'FFT': n_bins = len(XT_FFT[0,0,:]) # ou (fh-fl) * 4 # Número total de bins de frequencia 
    elif filtro == 'IIR': n_bins = fh-fl 
    step = int(n_bins / n_bands)
    size = int(step / overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
    
    # Make sub-bands limits and Temporal/Spectral filtering
    SCORE_T = np.zeros((len(XT), n_bands)) #(144, n_bands)
    SCORE_V = np.zeros((len(XV), n_bands))
    
    for i in range(n_bands):
        if filtro == 'FFT': # Only Spectral filtering
            bin_ini = i*step
            bin_fim = i*step+size
            #print(bin_ini,'-',bin_fim)
            if bin_fim >= n_bins: bin_fim = n_bins-1
            XTF = XT_FFT[:, :, bin_ini:bin_fim]
            XVF = XV_FFT[:, :, bin_ini:bin_fim]
            
        elif filtro == 'IIR': # Only Temporal filtering
            fl_sb = i*step + fl
            fh_sb = i*step + fl + size
            if fl_sb == 0: fl_sb = 0.001
            if fh_sb > fh: fh_sb = fh
            # print(fl_sb,'-',fh_sb)
            Wnl = fl_sb/nf
            Wnh = fh_sb/nf
            if Wnh >= 1: Wnh = 0.99
            
            b, a = butter(order, [Wnl, Wnh], btype='bandpass') # to filt IIR
            XTF = lfilter(b, a, XT) 
            XVF = lfilter(b, a, XV)
            
        csp = CSP(n_components=n_comp)
        csp.fit(XTF, y)
        XT_CSP = csp.transform(XTF) 
        XV_CSP = csp.transform(XVF)

        clf = LinearDiscriminantAnalysis()
        clf.fit(XT_CSP, y)
        SCORE_T[:, i] = np.ravel(clf.transform(XT_CSP)) # classificaçoes de cada época nas N sub bandas - auto validação
        SCORE_V[:, i] = np.ravel(clf.transform(XV_CSP)) # validação

    # Meta-classificador Bayesiano
    SCORE_T0 = SCORE_T[y == 0, :]
    m0 = np.mean(SCORE_T0, axis=0) #media classe A
    std0 = np.std(SCORE_T0, axis=0) #desvio padrão classe A
    
    SCORE_T1 = SCORE_T[y == 1, :]
    m1 = np.mean(SCORE_T1, axis=0)
    std1 = np.std(SCORE_T1, axis=0)

    p0 = norm(m0, std0) # p0 e p1 representam uma distribuição normal de médias m0 e m1, e desvio padrão std0 e std1
    p1 = norm(m1, std1)
    
    META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
    META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
    
    # SVM on top of the meta-classifier
    svc = SVC(kernel="linear", C=10**Clog)
    svc.fit(META_SCORE_T, y)
    scores = svc.predict(META_SCORE_V)
    
    cost = time() - start # stop timer (cost estimate)
    acc = np.mean(scores == y) 

    print ('A'+str(suj), cl, str(round(acc*100, 2))+'%')
    
    
    
    
    
   
    
    # path = '/mnt/dados/eeg_data/BCI4_2a/npy_all/A0' + str(suj) + '_all_'
    # events = np.load(path + 'events.npy')
    # data = np.load(path + 'data.npy')
    # data = data[:eeg_channels] # get only 22 EEG channels
    # epochs, labels = extractEpochs(data, events, smin, smax, cl)
    
    # X = epochs
    # y = labels
    
    # # epochs = [epochs[np.where(labels==i)] for i in cl]
    # # X = np.concatenate([epochs[0], epochs[1]])
    # # y = np.concatenate([np.zeros(len(epochs[0]), dtype=int), np.ones(len(epochs[1]), dtype=int)])
    
    # kf = StratifiedShuffleSplit(10, test_size=0.2, random_state=42)
    # kf = StratifiedKFold(10, False)

    # start = time()  # start timer
    # cross_scores = []
    # for idx_treino, idx_teste in kf.split(X, y):
    #     XT = X[idx_treino]
    #     XV = X[idx_teste]
    #     yT = y[idx_treino]
    #     yV = y[idx_teste]
        
    #     if filtro == 'FFT':  # FFT - Spectral filtering
    #         bmin = int(fl * (fs / nf))
    #         bmax = int(fh * (fs / nf))
    #         filtered = fft(XT)
    #         REAL = np.transpose(np.real(filtered)[:, :, bmin:bmax], (2, 0, 1))
    #         IMAG = np.transpose(np.imag(filtered)[:, :, bmin:bmax], (2, 0, 1))
    #         filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    #         XT_FFT = np.transpose(filtered, (1, 2, 0))
    #         filtered = fft(XV)
    #         REAL = np.transpose(np.real(filtered)[:, :, bmin:bmax], (2, 0, 1))
    #         IMAG = np.transpose(np.imag(filtered)[:, :, bmin:bmax], (2, 0, 1))
    #         filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    #         XV_FFT = np.transpose(filtered, (1, 2, 0))

    #     # Divide sub-bands
    #     if filtro == 'FFT':
    #         n_bins = len(XT_FFT[0, 0, :])  # ou (fh-fl) * 4 # Número total de bins de frequencia
    #     elif filtro == 'IIR':
    #         n_bins = fh - fl
    #     overlap = 2
    #     step = int(n_bins / n_bands)
    #     size = int(step * overlap)  # tamanho fixo p/ todas sub bandas. overlap em 50%

    #     # Make sub-bands limits and Temporal/Spectral filtering
    #     SCORE_T = np.zeros((len(XT), n_bands))
    #     SCORE_V = np.zeros((len(XV), n_bands))
    #     teste = 0
    #     for i in range(n_bands):
    #         if filtro == 'FFT': # Only FFT Spectral filtering
    #             bin_ini = i * step
    #             bin_fim = i * step + size
    #             if bin_fim >= n_bins: bin_fim = n_bins - 1
    #             XTF = XT_FFT[:, :, bin_ini:bin_fim]
    #             XVF = XV_FFT[:, :, bin_ini:bin_fim]
    #         elif filtro == 'IIR': # Only IIR Temporal filtering
    #             fl_sb = i * step + fl
    #             fh_sb = i * step + size + fl
    #             if fl_sb == 0: fl_sb = 0.001
    #             if fh_sb > fh: fh_sb = fh
    #             # print(fl_sb, fh_sb, nf, fl_sb/nf, fh_sb/nf)

    #             Wnl = fl_sb / nf
    #             Wnh = fh_sb / nf
    #             if Wnh >= 1: Wnh = 0.99

    #             b, a = sp.butter(order, [Wnl, Wnh], btype='bandpass')  # to filt IIR
    #             XTF = sp.lfilter(b, a, XT)  # comment here
    #             XVF = sp.lfilter(b, a, XV)  # comment here
                
    #         teste += XTF.mean() + XVF.mean()

    #         csp = CSP(n_components=n_comp)
    #         csp.fit(XTF, yT)
    #         XT_CSP = csp.transform(XTF)
    #         XV_CSP = csp.transform(XVF)

    #         clf = LinearDiscriminantAnalysis()
    #         clf.fit(XT_CSP, yT)
    #         SCORE_T[:, i] = np.ravel(clf.transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
    #         SCORE_V[:, i] = np.ravel(clf.transform(XV_CSP))  # validação
            
    #     SCORE_T0 = SCORE_T[yT == cl[0], :]
    #     m0 = np.mean(SCORE_T0, axis=0)  # media classe A
    #     std0 = np.std(SCORE_T0, axis=0)  # desvio padrão classe A

    #     SCORE_T1 = SCORE_T[yT == cl[1], :]
    #     m1 = np.mean(SCORE_T1, axis=0)
    #     std1 = np.std(SCORE_T1, axis=0)

    #     p0 = norm(m0, std0)  # p0 e p1 representam uma distribuição normal de médias m0 e m1, e desvio padrão std0 e std1
    #     p1 = norm(m1, std1)

    #     META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
    #     META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))

    #     svc = SVC(kernel="linear", C=10 ** Clog)

    #     # if classifier['clf']['type'] == 'LDA':
    #     #     svc = LDA()

    #     # if classifier['clf']['type'] == 'SVM':
    #     #     svc = SVC(kernel=classifier['clf']['kernel']['ktype'],
    #     #               C=10 ** (classifier['clf']['C']))

    #     # if classifier['clf']['type'] == 'KNN':
    #     #     svc = KNeighborsClassifier(n_neighbors=int(classifier['clf']['n_neighbors']),
    #     #                                metric='minkowski',
    #     #                                p=2)  # minkowski e p=2 -> para usar distancia euclidiana padrão

    #     # if classifier['clf']['type'] == 'DTree':
    #     #     svc = DecisionTreeClassifier(criterion=classifier['clf']['criterion'],
    #     #                                  max_depth=classifier['clf']['max_depth'],
    #     #                                  min_samples_split=math.ceil(classifier['clf']['min_samples_split']),
    #     #                                  random_state=0)  # None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados

    #     # if classifier['clf']['type'] == 'Bayes':
    #     #     svc = GaussianNB()

    #     # if classifier['clf']['type'] == 'MLP':
    #     #     svc = MLPClassifier(verbose=False,
    #     #                         max_iter=10000,
    #     #                         tol=0.0001,
    #     #                         activation=classifier['clf']['activation']['act_type'],
    #     #                         learning_rate_init=classifier['clf']['eta'],
    #     #                         learning_rate=classifier['clf']['eta_schedule']['eta_type'],
    #     #                         solver=classifier['clf']['solver']['solver_type'],
    #     #                         hidden_layer_sizes=(int(classifier['clf']['hidden_n_neurons']), int(classifier['clf']['n_hidden'])))

    #     svc.fit(META_SCORE_T, yT)
    #     scores = svc.predict(META_SCORE_V)
    #     cross_scores.append(np.mean(scores == yV))

    # cost = time() - start  # stop timer (cost estimate)
    # acc = np.mean(cross_scores)

    # print(round(acc*100,2)) 
    
    
    