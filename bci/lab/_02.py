#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:32:01 2020
@author: vboas
"""

import math
import itertools
import numpy as np
from scipy.linalg import eigh
from scipy.fftpack import fft
from bci_utils import labeling, extractEpochs
from scipy.signal import lfilter, butter, iirfilter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score
import pandas as pd
    
subjects = [1]
class_ids = [1, 2]

ACC = []
for suj in subjects:
    data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0'+str(suj)+'.npy', allow_pickle=True)
    Fs = info['fs']
    
    f_low, f_high = 0, 40
    ncsp = 8
    nbands = 9
    filtering = 'DFT'
        
    #%% EXTRACT TRIALS OF THE FULL EEG SIGNAL
    cond = False
    for cl in class_ids: cond += (events[:,1]==cl) 
    idx = np.ravel(np.where(cond))
    cues = events[idx,1]
    s0 = events[idx-1,0] # indice amostra que inicia cada trial 
    sn = s0 + math.floor(7 * Fs) # min(np.diff(s0))/Fs ## menor tempo entre os inicios de dois trials = 7.58s ou 1895 amostras
    trials = np.asarray([data[:, s0[i]:sn[i]] for i in range(len(s0))])
    
    #%% TRIALS SEGMENTATION: TRAIN, TEST, EVALUATE SETS
    # kf1 = StratifiedShuffleSplit(1, test_size=0.345, random_state=42)
    # for idx_train, idx_test in kf1.split(trials, cues): 
    #     trials_step1, cues_step1 = trials[idx_train], cues[idx_train]
    #     trials_val, t_val = trials[idx_test] , cues[idx_test]
        
    # kf2 = StratifiedShuffleSplit(1, test_size=0.465, random_state=42)
    # for idx_train, idx_test in kf2.split(trials_step1, cues_step1): 
    #     trials_train, t_train = trials_step1[idx_train], cues_step1[idx_train]
    #     trials_test , t_test =  trials_step1[idx_test] , cues_step1[idx_test]
        
    # kf2 = StratifiedShuffleSplit(1, test_size=0.5, random_state=42)
    # for idx_train, idx_test in kf2.split(trials, cues): 
    #     trials_train, t_train = trials[idx_train], cues[idx_train]
    #     trials_test , t_test =  trials[idx_test] , cues[idx_test]
    
    trials_train, t_train = trials[:len(cues)//2], cues[:len(cues)//2]
    trials_test , t_test =  trials[len(cues)//2:] , cues[len(cues)//2:]
    
    tmin, tmax = 2.5, 4.5
    
    windows0 = np.asarray([[i,f,f-i] for i in np.arange(2.5,4.5,0.5) for f in np.arange(i+1,6.5,0.5)])    
    windows0 = np.asarray([[2.5,4.5,2]])
    ACC_WIN0 = []
    for win0 in windows0:    
    
        #%% TRAINING    
        smin = math.floor(Fs * win0[0])
        smax = math.floor(Fs * win0[1])
        Z_train = trials_train[:,:,smin:smax]
        
        n_bins = f_high - f_low
        overlap = 0.5 
        step = n_bins / (nbands+1)
        size = step / overlap
        
        sub_bands = []
        for i in range(nbands):
            fl_sb = i * step + f_low
            fh_sb = i * step + size + f_low
            sub_bands.append([fl_sb, fh_sb])
        
        #%% Filtering
        if filtering == 'DFT':
            dft_resf = Fs/(smax-smin) 
            bin_size = 2/dft_resf # 2 representa sen e cos
            bmin = round(sub_bands[i][0] * bin_size)
            bmax = round(sub_bands[i][1] * bin_size)
            ZF = fft(Z_train)
            REAL = np.transpose(np.real(ZF), (2, 0, 1))
            IMAG = np.transpose(np.imag(ZF), (2, 0, 1))
            ZF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
            ZF = np.transpose(ZF, (1, 2, 0))
            X_train = [ ZF[:, :, round(sub_bands[i][0]*bin_size):round(sub_bands[i][1]*bin_size)] for i in range(nbands) ]
              
        elif filtering == 'IIR':
            nyq = 0.5 * Fs
            X_train = []
            for i in range(nbands):
                low = sub_bands[i][0] / nyq
                high = sub_bands[i][1] / nyq
                if low == 0: low = 0.001
                if high >= 1: high = 0.99
                b, a = butter(5, [low,high], btype='bandpass')
                X_train.append(lfilter(b, a, Z_train)) 
                # b, a = iirfilter(5, [low,high], btype='band')
                # X_train.append(filtfilt(b, a, Z_train)) 
        
        #%% CSP
        filters_ = []
        for i in range(nbands):
            e, c, s = X_train[i].shape
            classes = np.unique(t_train)   
            Xa = X_train[i][classes[0] == t_train,:,:]
            Xb = X_train[i][classes[1] == t_train,:,:]
            
            Sa = np.zeros((c, c)) 
            Sb = np.zeros((c, c))
            for i in range(int(e/2)):
                # Sa += np.dot(Xa[i,:,:], Xa[i,:,:].T)
                # Sb += np.dot(Xb[i,:,:], Xb[i,:,:].T)
                Sa += np.dot(Xa[i,:,:], Xa[i,:,:].T) / Xa[i].shape[-1] # sum((Xa * Xa.T)/q)
                Sb += np.dot(Xb[i,:,:], Xb[i,:,:].T) / Xb[i].shape[-1] # sum((Xb * Xb.T)/q)
            Sa /= len(Xa)
            Sb /= len(Xb)
            
            [D, W] = eigh(Sa, Sa + Sb)
            ind = np.empty(c, dtype=int)
            ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
            ind[1::2] = np.arange(0, c // 2)
            W = W[:, ind]
            filters_.append(W.T[:ncsp])
        
        Y_train = [ np.asarray([ np.dot(filters_[i], ep) for ep in X_train[i] ]) for i in range(nbands)] # Spatial Filtering
        XCSP_train = [ np.log(np.mean(Y_train[i] ** 2, axis=2)) for i in range(nbands) ] # Feature extraction
        # XCSP_train = [ np.log(np.var(Y_train[i], axis=2)) for i in range(nbands) ] # Feature extraction
            
        #%% LDA (SUB-BAND SCORES)
        lda = [ LDA() for i in range(nbands)]
        for i in range(nbands): lda[i].fit(XCSP_train[i], t_train)
        
        SCORE_train = np.asarray([ np.ravel(lda[i].transform(XCSP_train[i])) for i in range(nbands) ]).T 
        
        # classificações de cada época nas N sub bandas - auto validação
        # O LDA transforma os dados de treino e validação em cada sub-banda onde o resultado contínuo do LDA é mantido para construir um vetor de características de dimensão (n_bands)    
               
        #%% Bayesian Meta-Classifier
        SCORE_A = SCORE_train[t_train == class_ids[0], :]
        SCORE_B = SCORE_train[t_train == class_ids[1], :] # separa os scores de cada sub-banda referentes a classe A e B
        p0 = norm(np.mean(SCORE_A, axis=0), np.std(SCORE_A, axis=0)) # projeta uma distribuição normal/gaussiana a partir das médias (m0,m1) e dp (std0,std1) dos scores LDA em cada sub-banda ... 
        p1 = norm(np.mean(SCORE_B, axis=0), np.std(SCORE_B, axis=0)) # entre todas as épocas de cada classe. p0 é uma espécie de filtro bayesiano favorável à classe A e p1 à classe B
        META_SCORE_train = np.log(p0.pdf(SCORE_train) / p1.pdf(SCORE_train)) # scores aplicados nas funções de densidade de probabilidade de cada classe e calculada a razão entre elas.
        # META_SCORE > 0 indica época melhor representada pela função p0
        # META_SCORE < 0 a melhor representação é por p1
        
        #%% Final classification   
        clf_final = SVC(kernel='linear', C=1e-4, probability=True)
        # clf_final = LDA()
        # clf_final = GaussianNB()
        # clf_final = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)           
        # clf_final = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=None, min_samples_split=2)
        # clf_final = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4, learning_rate_init=0.001, alpha=0.0001, activation='relu', hidden_layer_sizes=(100,2))
        clf_final.fit(META_SCORE_train, t_train)
        
        #%% TESTING
        L = (tmax-tmin)
        a = 0.25
        windows = np.asarray([[i,i+L] for i in np.arange(0,(7+a)-L,a)])
        # windows = np.asarray([[i,f,f-i] for i in np.arange(0,6.5,0.5) for f in np.arange(i+L,7.5,0.5)])
        
        acc_win_test, kappa_suj, PY, Y = [], [], [], []
        for win in windows:
            smin = math.floor(Fs * win[0])
            smax = math.floor(Fs * win[1])
            
            # smin = math.floor(Fs * 2.5)
            # smax = math.floor(Fs * 4.5)
            Z_test = trials_test[:,:,smin:smax]
            
            if filtering == 'DFT':
                dft_resf = Fs/(smax-smin) 
                bin_size = 2/dft_resf # 2 representa sen e cos
                bmin = round(sub_bands[i][0] * bin_size)
                bmax = round(sub_bands[i][1] * bin_size)
                ZF = fft(Z_test)
                REAL = np.transpose(np.real(ZF), (2, 0, 1))
                IMAG = np.transpose(np.imag(ZF), (2, 0, 1))
                ZF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
                ZF = np.transpose(ZF, (1, 2, 0))
                X_test = [ ZF[:, :, round(sub_bands[i][0]*bin_size):round(sub_bands[i][1]*bin_size)] for i in range(nbands) ]
                  
            elif filtering == 'IIR':
                nyq = 0.5 * Fs
                X_test = []
                for i in range(nbands):
                    low = sub_bands[i][0] / nyq
                    high = sub_bands[i][1] / nyq
                    if low == 0: low = 0.001
                    if high >= 1: high = 0.99
                    b, a = butter(5, [low,high], btype='bandpass')
                    X_test.append(lfilter(b, a, Z_test)) 
                    # b, a = iirfilter(5, [low,high], btype='band')
                    # X_test.append(filtfilt(b, a, Z_test)) 
                    
            Y_test = [ np.asarray([ np.dot(filters_[i], ep) for ep in X_test[i] ]) for i in range(nbands) ] # Spatial Filtering
            
            XCSP_test = [ np.log(np.mean(Y_test[i]**2, axis=2)) for i in range(nbands) ] # Feature extraction
            # XCSP_test = [ np.log(np.var(Y_test[i], axis=2)) for i in range(nbands) ] # Feature extraction
            
            score = np.dot(XCSP_test[0] - lda[0].xbar_, lda[0].scalings_)
            
            SCORE_test = np.asarray([ np.ravel(lda[i].transform(XCSP_test[i])) for i in range(nbands) ]).T # Score LDA
                
            META_SCORE_test = np.log(p0.pdf(SCORE_test) / p1.pdf(SCORE_test)) # Bayesian Meta-score
            
            y = META_SCORE_test @ clf_final.coef_.T # y<0=classe A(1) ; y>0=classe B(2)
            y_labels = clf_final.predict(META_SCORE_test) # Final classification
            y_proba = clf_final.predict_proba(META_SCORE_test)
            
            Y.append(y_labels)
            PY.append(y_proba)
            # print(win[0], win[1], list(map(lambda x: round(x,2), y_proba[1])), y_labels[1], t_test[1])
            
            
            acc = clf_final.score(META_SCORE_test, t_test)
            acc = np.mean(y_labels == t_test)
            kappa = cohen_kappa_score(y_labels, t_test)
            acc_win_test.append(acc)
            print(win[0], win[1], round(acc*100,2))
            print(win[0], win[1], list(map(lambda x: round(x,2), y_proba[1])), y_labels[1], t_test[1])
            # print('Mean proba: ', round(np.mean(y_proba, axis=0)[0]/np.mean(y_proba, axis=0)[1],2))
            # print('kappa:', round(kappa,4))
        
        acc_mean_test = round(np.mean(acc_win_test)*100,2)
        
        Y = np.transpose(Y)
        
        PY = np.transpose(PY, (1,2,0))
        
        PY = np.mean(PY, axis=2)
        
        Y_ = ([np.argmax(py) for py in PY] + np.ones(len(PY))).astype(int)
        
        acc_mean_test_probs = round(np.mean(Y_ == t_test)*100,2)
        
        print([win0[0], win0[1], acc_mean_test, acc_mean_test_probs])
        ACC_WIN0.append([win0[0], win0[1], acc_mean_test, acc_mean_test_probs])
        
    ACC = pd.DataFrame(np.c_[np.asarray(ACC_WIN0)], columns=['Jini','Jfim','Mean_Acc','Mean_Prob'])

    # best_win_suj = np.c_[np.asarray([ np.ravel(ACC[ACC[col] == R.max()][col].iloc[0]) for col in R.columns[3:] ]),
    #                  np.asarray([ np.ravel(R[R[col] == R[col].max()].iloc[0,:3]) for col in R.columns[3:] ])]


# smin, smax = math.floor(0.5 * Fs), math.floor(2.5 * Fs)
# epochsT, labelsT = extractEpochs(d_train, e_train, smin, smax, class_ids)
# epochsV, labelsV = extractEpochs(d_test, e_test, smin, smax, class_ids)

# ZT = [epochsT[np.where(labelsT==i)] for i in class_ids]
# ZT = np.r_[ZT[0],ZT[1]]
# tT = np.r_[class_ids[0]*np.ones(int(len(ZT)/2)), class_ids[1]*np.ones(int(len(ZT)/2))]


# f_low, f_high = 8, 30
# DFT = 1 # 0=IIR, 1=DFT

# if DFT:
#     buffer_len = smax - smin
#     dft_res_freq = Fs/buffer_len # resolução em frequência fft
#     dft_size_band = round(2/dft_res_freq) # 2 representa sen e cos que foram separados do componente complexo da fft intercalados
    
#     data_out = fft(ZT)
#     REAL = np.transpose(np.real(data_out), (2, 0, 1))
#     IMAG = np.transpose(np.imag(data_out), (2, 0, 1))
#     data_out = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
#     XT_FFT = np.transpose(data_out, (1, 2, 0))

#     bmin = f_low * dft_size_band
#     bmax = f_high * dft_size_band
#     # print(bmin, bmax)
#     XT = XT_FFT[:, :, bmin:bmax]

# else: # IIR Filtering
#     nyq = 0.5 * Fs
#     low = f_low / nyq
#     high = f_high / nyq
#     if low == 0: low = 0.001
#     if high >= 1: high = 0.99
#     b, a = butter(5, [low, high], btype='bandpass')
#     # b, a = iirfilter(5, [low,high], btype='band')
#     # XT = lfilter(b, a, ZT) 
#     XT = filtfilt(b, a, ZT)

# #%% CSP
# ncomp = 8
# e, c, s = XT.shape
# classes = np.unique(tT)   
# Xa = XT[classes[0] == tT,:,:]
# Xb = XT[classes[1] == tT,:,:]

# Sa = np.zeros((c, c)) 
# Sb = np.zeros((c, c))
# for i in range(int(e/2)):
#     # Sa += np.dot(Xa[i,:,:], Xa[i,:,:].T)
#     # Sb += np.dot(Xb[i,:,:], Xb[i,:,:].T)
#     Sa += np.dot(Xa[i,:,:], Xa[i,:,:].T) / Xa[i].shape[-1] # sum((Xa * Xa.T)/q)
#     Sb += np.dot(Xb[i,:,:], Xb[i,:,:].T) / Xb[i].shape[-1] # sum((Xb * Xb.T)/q)
# Sa /= len(Xa)
# Sb /= len(Xb)

# [D, W] = eigh(Sa, Sa + Sb)
# ind = np.empty(c, dtype=int)
# ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
# ind[1::2] = np.arange(0, c // 2)
# W = W[:, ind]
# Wf = W.T[:ncomp]
       
# YT = np.asarray([np.dot(Wf, ep) for ep in XT])

# #%% Feature extraction
# XT_CSP = np.log(np.mean(YT ** 2, axis=2))
# # XT_CSP = np.log(np.var(YT, axis=2))

# #%% LDA Classifier
# clf = LDA()
# clf.fit(XT_CSP, tT)

# ############################################################
# #%% ################# EVALUATE UNIQUE EPOCH
# ############################################################

# for ep in range(1):
#     idx = int(np.random.choice(epochsV.shape[0], 1)) # escolhe uma época de teste
#     Z, t = epochsV[idx], labelsV[idx]
    
#     #%%  Filtering
#     if DFT:
#         data_out = fft(Z)
#         REAL = np.real(data_out).T
#         IMAG = np.imag(data_out).T
#         XFFT = np.transpose(list(itertools.chain.from_iterable(zip(IMAG, REAL))))
#         bmin = f_low * dft_size_band
#         bmax = f_high * dft_size_band
#         # print(bmin, bmax)
#         X = XFFT[:, bmin:bmax]
#     else:
#         nyq = 0.5 * Fs
#         low = f_low / nyq
#         high = f_high / nyq
#         if high >= 1: high = 0.99
#         b, a = butter(5, [low, high], btype='bandpass')
#         # b, a = iirfilter(5, [low,high], btype='band')
#         # X = lfilter(b, a, ZT) 
#         X = filtfilt(b, a, ZT)
    
#     #%% CSP
#     Y = np.dot(Wf, X)
#     features = np.log(np.mean(Y**2, axis=1))
#     # features = np.log(np.var(Y, axis=1))
    
#     #%% LDA Classifier
#     y_label = clf.predict(features.reshape(1, -1))
#     y_prob = clf.predict_proba(features.reshape(1, -1))
    
#     print(f'Epoch idx: {idx}\nTrue target (t): {t}\nPredicted target (y): {y_label}\nLikely: {y_label==t}\nClasses Prob: {y_prob}')