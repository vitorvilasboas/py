# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import os
import mne
import itertools
import math
import numpy as np
import pandas as pd
from time import time
from sklearn.svm import SVC
from scipy.io import loadmat
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.linalg import eigh
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold

""" 9 subjects | 2 classes (LH, RH) | 3 channels | Fs 250Hz
    6 channels (first 3 is EEG: C3, C4, Cz; last 3 is EOG)
    120 trials (60 per class) - 5 sessions
    2 sessions without feedback
    3 sessions with feedback (smiley)
    Total sessions = 5 (01T,02T,03T,04E,05E)
         5 * 120 trials = 600 total trials -> 5*60 = 300 per class -> 2*60 = 120 per session
	     2 training sessions (no feedback) - 01T,02T 
         1 training session (WITH feedback) - 03T
	     2 evaluate sessions (WITH feedback) - 04E,05E 
"""

def corrigeNaN(data):
    for ch in range(data.shape[0] - 1):
        this_chan = data[ch]
        data[ch] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(data[ch])
        meanChannel = np.nanmean(data[ch])
        data[ch, mask] = meanChannel
    return data

def nanCleaner(epoch):
    """Removes NaN from data by interpolation
    data_in : input data - np matrix channels x samples
    data_out : clean dataset with no NaN samples"""
    for i in range(epoch.shape[0]):
        bad_idx = np.isnan(epoch[i, :])
        epoch[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoch[i, ~bad_idx])
    return epoch

def labeling_lnx(ev, ds, suj, trues): # normaliza rotulos de eventos conforme descrição oficial do dataset
    """ Dataset Description (by vboas): more info in http://bbci.de/competition/iv/desc_2b.pdf
        01T e 02T (without feedback)
    		10 trial * 2 classes * 6 runs * 2 sessions = 240 trials (120 per class)
    		Cross t=0 (per 3s)
    		beep t=2s
    		cue t=3s (per 1.25s)
    		MI t=4s (per 3s)
    		Pause t=7s (per 1.5-2.5s)
    		EndTrial t=8.5-9.5
    	03T, 04E e 05E (with feedback)
    		10 trial * 2 classes * 4 runs * 3 sessions = 240 trials (120 per class)
    		Smiley(grey) t=0 (per 3.5s)
    		beep t=2s
    		cue t=3s (per 4.5s)
    		MI (Feedback períod) t=3.5s (per 4s)
    		Pause t=7.5s (per 1-2s)
    		EndTrial t=8.5-9.5
    	Meta-info 01T e 02T:
    		1=1023 (rejected trial)
    		2=768 (start trial)
    		3=769 (Class 1 - LH - cue onset)
    		4=770 (Class 2 - RH - cue onset)
    		5=277 (Eye closed)
    		6=276 (Eye open)
    		7=1081 (Eye blinks)
    		8=1078 (Eye rotation)
    		9=1077 (Horizontal eye movement)
    		10=32766 (Start a new run) *(to B0102T == 5)
    		11=1078 (Vertical eye movement)			
    	Meta-info 03T:
    		1=781 (BCI feedback - continuous)
    		2=1023 (rejected trial)
    		3=768 (start trial)
    		4=769 (Class 1 - LH - cue onset)
    		5=770 (Class 2 - RH - cue onset)
    		6=277 (Eye closed)
    		7=276 (Eye open)
    		8=1081 (Eye blinks)
    		9=1078 (Eye rotation)
    		10=1077 (Horizontal eye movement)
    		11=32766 (Start a new run)
    		12=1078 (Vertical eye movement)
    	Meta-info 04E e 05E:
    		1=781 (BCI feedback - continuous)
    		2=1023 (rejected trial)
    		3=768 (start trial)
    		4=783 (Cue unknown/undefined)
    		5=277 (Eye closed)
    		6=276 (Eye open)
    		7=1081 (Eye blinks)
    		8=1078 (Eye rotation)
    		9=1077 (Horizontal eye movement)
    		10=32766 (Start a new run)
    		11=1078 (Vertical eye movement)
    """
    # Remove marcações inúteis e normaliza rotulos de eventos conforme descrição oficial do dataset
    if ds in ['01T','02T']:
        for rm in range(5,12): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
        ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete rejected trials
        ev[:,1] = np.where(ev[:,1]==2, 0, ev[:,1]) # altera label start trial de 2 para 0
        ev[:,1] = np.where(ev[:,1]==3, 1, ev[:,1]) # altera label cue LH de 3 para 1
        ev[:,1] = np.where(ev[:,1]==4, 2, ev[:,1]) # altera label cue RH de 4 para 2
        for i in range(len(ev[:,1])):
            if ev[i,1]==0: ev[i,1] = ev[i+1,1] + 768
    elif ds=='03T': 
        for rm in range(6,13): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
        ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # delete rejected trials
        ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete feedback continuous
        ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1]) # altera label start trial de 3 para 0
        ev[:,1] = np.where(ev[:,1]==4, 1, ev[:,1]) # altera label cue LH de 4 para 1
        ev[:,1] = np.where(ev[:,1]==5, 2, ev[:,1]) # altera label cue RH de 5 para 2
        for i in range(len(ev[:,1])):
            if ev[i,1]==0: ev[i,1] = ev[i+1,1] + 768
    else:
        for rm in range(5,12): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
        ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # delete rejected trials
        ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete feedback continuous
        ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1]) # altera label start trial de 3 para 0
        ev[np.where(ev[:,1]==4),1] = trues #rotula momento da dica conforme truelabels
        for i in range(len(ev[:,1])):
            if ev[i,1]==0: ev[i,1] = ev[i+1,1] + 768
    return ev

def load_gdf_info(path, suj, ds, eeg_channels):
    mne.set_log_level('WARNING','DEBUG')
    raw = mne.io.read_raw_gdf(path + 'B0' + str(suj) + ds + '.gdf')
    raw.load_data()
    data = raw.get_data() # [channels x samples]
    data = data[:eeg_channels]
    data = corrigeNaN(data) # Correção de NaN nos dados brutos
    events = raw.find_edf_events()
    timestamps = np.delete(events[0],1,axis=1) # elimina coluna de zeros
    truelabels = np.ravel(loadmat(path + 'true_labels/B0' + str(suj) + ds + '.mat')['classlabel'])
    events = labeling_lnx(timestamps, ds, suj, truelabels) # Labeling correctly the events like competition description
    # timestamps[:,1] = labeling_mac_win(timestamps[:,1], ds, suj, truelabels) # Labeling correctly the events like competition description
    return data, timestamps

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
    
def csp_lda(args):
    XT, XV, y, filtro, order, fl, fh, fs, nf, n_comp, n_bands, Clog = args
    if filtro == 'FFT': # Spectral filtering
        filtered = fft(XT)
        bin0 = int(fl * (fs/nf))
        binN = int(fh * (fs/nf))
        REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1)) 
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        XTF = np.transpose(filtered, (1, 2, 0))       
        filtered = fft(XV)
        REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1))
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
        XVF = np.transpose(filtered, (1, 2, 0))
    elif filtro == 'IIR': # Temporal filtering
        if fl==0: fl=0.001
        b, a = butter(order, [fl/nf, fh/nf], btype='bandpass') # to filt IIR
        XTF = lfilter(b, a, XT)
        XVF = lfilter(b, a, XV)
    csp = CSP(n_components=n_comp)
    csp.fit(XTF, y)
    XT_CSP = csp.transform(XTF)
    XV_CSP = csp.transform(XVF) 
    clf = LinearDiscriminantAnalysis()  
    #clf = SVC(kernel="poly", C=10**(-4))
    #clf = KNeighborsClassifier(n_neighbors=24, metric='minkowski', p=2) #minkowski e p=2 -> para usar distancia euclidiana padrão
    #clf = DecisionTreeClassifier(criterion='entropy', random_state=0) #max_depth = None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
    #clf = GaussianNB()
    #clf = MLPClassifier(verbose=False, max_iter=10000, tol=0.0001, activation='logistic', learning_rate_init=0.001, learning_rate='invscaling',  solver='adam') #hidden_layer_sizes=(100,),
    clf.fit(XT_CSP, y)
    scores = clf.predict(XV_CSP)
    acc = np.mean(scores == y) 
    return acc

def sbcsp(args):
    XT, XV, y, filtro, order, fl, fh, fs, nf, n_comp, n_bands, Clog = args
    if filtro == 'FFT':
        # Spectral filtering
        filtered = fft(XT)
        bin0 = int(fl * (fs/nf))
        binN = int(fh * (fs/nf))
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
    overlap = 2
    step = int(n_bins / n_bands)
    size = int(step * overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
    
    # Make sub-bands limits and Temporal/Spectral filtering
    SCORE_T = np.zeros((len(XT), n_bands))
    SCORE_V = np.zeros((len(XV), n_bands))
    for i in range(n_bands):
        if filtro == 'FFT':
            # Only Spectral filtering
            bin_ini = i*step
            bin_fim = i*step+size
            if bin_fim >= n_bins: bin_fim = n_bins-1
            XTF = XT_FFT[:, :, bin_ini:bin_fim]
            XVF = XV_FFT[:, :, bin_ini:bin_fim]
        elif filtro == 'IIR':
        # Only Temporal filtering
            fl_sb = i*step+fl
            fh_sb = i*step+size+fl
            if fl_sb == 0: fl_sb = 0.001
            if fh_sb > fh: fh_sb = fh
            #print(fl_sb, fh_sb, nf, fl_sb/nf, fh_sb/nf)
            
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
    return np.mean(scores == y) 
                       
if __name__ == '__main__': # load GDF and create NPY files
    path = '/mnt/dados/eeg_data/BCI4_2b/'
    dataset = ['01T','02T','03T','04E','05E']
    subjects = range(1,10)
    classes = [1, 2]
    eeg_channels = 3
    fs = 250.0
    nf = fs/2.
    Tmin, Tmax = 3.5, 6.5 # startTrial=0; cue=3; startMI=4; endMI=7; endTrial=8.5-9.5
    smin = int(math.floor(Tmin * fs)) # initial sample (ex. 0)
    smax = int(math.floor(Tmax * fs)) # final sample (ex. 1875)
    filtro = 'IIR' #'IIR' or 'FFT'
    order = 5
    fl = 8
    fh = 30
    n_comp = 8
    n_bands = 15
    Clog = -3 
    
    ### uncomment to create npy files 
    # for suj in subjects:
    #     DT, EV = [], []
    #     for ds in dataset:
    #         data, events = load_gdf_info(path+'gdf/', suj, ds, eeg_channels)
    #         # Save separated npy files to each dataset
    #         np.save(path + 'npy/B0' + str(suj) + ds + '_data', data)
    #         np.save(path + 'npy/B0' + str(suj) + ds + '_events', events)
    #         DT.append(data)
    #         EV.append(events)
    #     # Save a unique npy file with all datasets
    #     soma = 0
    #     for i in range(1,len(EV)): 
    #         soma += len(DT[i-1].T)
    #         EV[i][:,0] += soma
    #     np.save(path + 'npy/B0' + str(suj) + '_all_data', np.c_[DT[0],DT[1],DT[2],DT[3],DT[4]])    
    #     np.save(path + 'npy/B0' + str(suj) + '_all_events', np.r_[EV[0],EV[1],EV[2],EV[3],EV[4]])
        
    RES = []
    for suj in subjects: 
        ## Usando npy treino e teste separados
        # for ds in ['01T','02T']:
        #     events = np.load(path + 'npy/B0' + str(suj) + ds + '_events.npy')
        #     data = np.load(path + 'npy/B0' + str(suj) + ds + '_data.npy')
        #     data = data[:eeg_channels] # get only 22 EEG channels
        #     epochs, labels = extractEpochs(data, events, smin, smax, classes)
        #     # epochs = nanCleaner(epochs) # Correção de NaN nas épocas
        #     if ds=='01T': XT = [ epochs[np.where(labels==i)] for i in classes] # Extrair épocas de cada classe
        #     if ds=='02T': XV = [ epochs[np.where(labels==i)] for i in classes] 
        
        ## usando npy único
        events = np.load(path + 'npy/B0' + str(suj) + '_all_events.npy')
        data = np.load(path + 'npy/B0' + str(suj) + '_all_data.npy')
        epochs, labels = extractEpochs(data, events, smin, smax, classes)
        # epochs = nanCleaner(epochs) # Correção de NaN nas épocas
        metade = int(len(epochs)/2)
        epochsT, labelsT = epochs[:metade], labels[:metade]
        epochsV, labelsV = epochs[metade:], labels[metade:]
        XT = [ epochsT[np.where(labelsT==i)] for i in classes] # Extrair épocas de cada classe
        XV = [ epochsV[np.where(labelsV==i)] for i in classes] # Extrair épocas de cada classe

        XT = np.concatenate([XT[0],XT[1]]) # Dados de treinamento das classes A, B
        XV = np.concatenate([XV[0],XV[1]]) # Dados de validação das classes A, B
        y = np.concatenate([np.zeros(int(len(XT)/2)), np.ones(int(len(XT)/2))]) # Vetor gabarito
        start = time() # start timer
        args = XT, XV, y, filtro, order, fl, fh, fs, nf, n_comp, n_bands, Clog
        acc = csp_lda(args)
        #acc = sbcsp(args)
        cost = time() - start # stop timer (cost estimate)
        RES.append([suj,acc,cost])
        print(suj, str(round(acc*100,2))+'%', str(int(cost*1000))+'ms')
    RES = np.array(RES)  
    print(f'{str(round(RES[:,2].mean()*100,2))}%  +- {str(round(RES[:,2].std()*100,1))}% || {round(RES[:,3].sum(),1)}s \n\n')
    