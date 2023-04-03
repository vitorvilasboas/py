# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import mne
import math
import itertools
import numpy as np
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
from sklearn.pipeline import Pipeline

""" 2 subjects (TL, WL) | 2 classes (lh, rh) | Fs 250Hz
    40 trials (20 per class) - TL: 2 sessions; WL:3 sessions
    8 channels (1=Cz 2=Cpz 3=C1 4=C3 5=CP3 6=C2 7=C4 8=CP4)
    Scalp map:      C3  C1  Cz  C2  CP4     	4  3  1  6  7
                       CP3   CPz  CP4 		  5   2   8
"""

def labeling_lnx(ev, ds): # normaliza rotulos de eventos conforme descrição oficial do dataset
    """ Dataset description/Meta-info MNE (Linux) (by vboas):
        1=Cross on screen (BCI experiment) 
        2=Feedback (continuous) - onset (BCI experiment)
        3=768 Start of Trial, Trigger at t=0s
        4=783 Unknown
        5=769 class1, Left hand - cue onset (BCI experiment)
        6=770 class2, Right hand - cue onset (BCI experiment)
    """
    ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # elimina marcações inuteis (cross on screen)
    ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # elimina marcações inuteis (feedback continuous)
    ev = np.delete(ev,np.where(ev[:,1]==4),axis=0) # elimina marcações inuteis (unknown)
    ev[:,1] = np.where(ev[:,1]==5, 769, ev[:,1]) # altera label lh de 5 para 1
    ev[:,1] = np.where(ev[:,1]==6, 770, ev[:,1]) # altera label rh de 6 para 2
    
    for i in range(len(ev)): 
        if ev[i,1]==3: ev[i,1] = ev[i+1,1] - 768 # rotula dica conforme tarefa (idx+1=769 ou 770, idx=1 ou 2)

    return ev

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

def load_gdf_info(path, ds, eeg_channels):
    mne.set_log_level('WARNING','DEBUG')
    raw = mne.io.read_raw_gdf(path + ds + '.gdf')
    raw.load_data()
    data = raw.get_data() # [channels x samples]
    data = data[:eeg_channels]
    data = corrigeNaN(data) # Correção de NaN nos dados brutos
    events_raw = raw.find_edf_events()
    events = np.delete(events_raw[0],1,axis=1) # elimina coluna de zeros
    events = labeling_lnx(events, ds) # Labeling correctly the events like competition description
    return data, events

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

def csp_lda_cross(args):
    X, y, filtro, order, fl, fh, fs, nf, n_comp, n_bands, Clog, folds, test_size = args
    if filtro == 'FFT': # Spectral filtering
        filtered = fft(X)
        bin0 = int(fl * (fs/nf))
        binN = int(fh * (fs/nf))
        REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1)) 
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        XF = np.transpose(filtered, (1, 2, 0))       
    elif filtro == 'IIR': # Temporal filtering
        if fl==0: fl=0.001
        b, a = butter(order, [fl/nf, fh/nf], btype='bandpass') # to filt IIR
        XF = lfilter(b, a, X)
    clf = LinearDiscriminantAnalysis()
    #clf = SVC(kernel="poly", C=10**(-4))
    #clf = KNeighborsClassifier(n_neighbors=24, metric='minkowski', p=2) #minkowski e p=2 -> para usar distancia euclidiana padrão
    #clf = DecisionTreeClassifier(criterion='entropy', random_state=0) #max_depth = None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
    #clf = GaussianNB()
    #clf = MLPClassifier(verbose=False, max_iter=10000, tol=0.0001, activation='logistic', learning_rate_init=0.001, learning_rate='invscaling',  solver='adam') #hidden_layer_sizes=(100,),
    
    ## Cross-validation: Option 1 (compact code)
    process = Pipeline([('CSP', CSP(n_components=n_comp)), ('classifier', clf)]) # run a process sequence csp + clf
    cv = StratifiedShuffleSplit(folds, test_size=test_size, random_state=1) # random_state=42 keep fixed the train and test indexes
    scores = cross_val_score(process, XF, y, cv=cv)
    acc = np.mean(scores)
    # print(scores)
    ## Cross-validation: Option 2
    # acc_iter = []
    # kf = StratifiedKFold(folds, False, 0)
    # for train_index, test_index in kf.split(X, y):
    #     XT = X[train_index]
    #     XV = X[test_index]
    #     csp = CSP(n_components=n_comp)
    #     csp.fit(XT, y[train_index])
    #     XT_CSP = csp.transform(XT)
    #     XV_CSP = csp.transform(XV)
    #     clf.fit(XT_CSP, y[train_index])
    #     scores = clf.predict(XV_CSP)
    #     acc_iter.append(np.mean(scores == y[test_index]))
    # acc = np.mean(acc_iter)
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
       
if __name__ == "__main__":
    path = '/mnt/dados/eeg_data/BCI_LINCE/'
    subjects = ['TL','WL']
    dataset = ['_S1','_S2']
    classes = [1,2]
    eeg_channels = 8
    fs = 250.0
    nf = fs/2.
    Tmin, Tmax = 3.5, 6.5 # Start trial= 0; Wait beep= 2; Start cue=3; Start MI= 4.25; End MI= 8; End trial(break)= 10-12
    smin = int(math.floor(Tmin * fs)) # initial sample (ex. 0)
    smax = int(math.floor(Tmax * fs)) # final sample (ex. 2500)
    filtro = 'IIR' #'IIR' or 'FFT'
    order = 5
    fl = 8
    fh = 30
    n_comp = 6
    n_bands = 15
    Clog = -3
    folds = 6
    test_size = 0.2
    
    ### uncomment to create npy files
    gdf_file = ['TL_S1','TL_S2','WL_S1','WL_S2']#,'WL_S3']
    # for file in gdf_file: # only first run
    #     data, events = load_gdf_info(path+'gdf/', file, eeg_channels)
    #     # Save separated npy files to each dataset type (train and train)
    #     np.save(path + 'npy_sep/' + file + '_data', data)
    #     np.save(path + 'npy_sep/' + file + '_events', events)
    # for suj in subjects:
    #     # Save a unique npy file with both dataset type (train + test)
    #     train_data = np.load(path + 'npy_sep/' + suj + '_S1_data.npy')
    #     test_data = np.load(path + 'npy_sep/' + suj + '_S2_data.npy')
    #     train_ev = np.load(path + 'npy_sep/' + suj + '_S1_events.npy')
    #     test_ev = np.load(path + 'npy_sep/' + suj + '_S2_events.npy')
    #     test_ev[:,0] += len(train_data.T) # pos de events_test + última pos de data_train (events_test dá continuidade a events_train)
    #     np.save(path + 'npy_full/' + suj + '_data', np.c_[train_data, test_data])    
    #     np.save(path + 'npy_full/' + suj + '_events', np.r_[train_ev, test_ev])
    
    for suj in subjects:
        # Use if epochs already extracted in npy files
        # X1 = list(np.load('/mnt/dados/eeg_data/eeg_epochs/BCI_CAMTUC/' + suj + '_S1.npy'))
        # X2 = list(np.load('/mnt/dados/eeg_data/eeg_epochs/BCI_CAMTUC/' + suj + '_S2.npy'))
        # XT = [ X1[i][:,:,smin:smax] for i in range(len(X1)) ]
        # XV = [ X2[i][:,:,smin:smax] for i in range(len(X2)) ]
        for ds in dataset:
            events = np.load(path + 'npy/' + suj + ds + '_events.npy')
            data = np.load(path + 'npy/' + suj + ds + '_data.npy')
            epochs, labels = extractEpochs(data, events, smin, smax, classes)
            # epochs = nanCleaner(epochs) # Correção de NaN nas épocas
            if ds=='_S1': XT = [ epochs[np.where(labels==i)] for i in classes] # Extrair épocas específicas de cada classe
            elif ds=='_S2': XV = [ epochs[np.where(labels==i)] for i in classes]
        
    ##### SINGLE VALIDATION:
        XT1 = np.concatenate([XT[0],XT[1]]) # Dados de treinamento das classes A, B
        XV1 = np.concatenate([XV[0],XV[1]]) # Dados de validação das classes A, B
        y = np.concatenate([np.zeros(int(len(XT1)/2)), np.ones(int(len(XT1)/2))]) # Vetor gabarito
        start = time() # start timer
        args = XT1, XV1, y, filtro, order, fl, fh, fs, nf, n_comp, n_bands, Clog
        acc = csp_lda(args) # sbcsp(args)
        cost = time() - start # stop timer (cost estimate)
        print(suj, str(round(acc*100,2))+'%', str(int(cost*1000))+'ms')
    
    #### CROSS-VALIDATION:
        # Xa = np.r_[XT[0],XV[0]] # LH
        # Xb = np.r_[XT[1],XV[1]] # RH
        # X = np.r_[Xa,Xb]
        data = np.load(path + 'npy/' + suj + '_all_data.npy')
        events = np.load(path + 'npy/' + suj + '_all_events.npy')
        epochs, labels = extractEpochs(data, events, smin, smax, classes)
        # epochs = nanCleaner(epochs) # Correção de NaN nas épocas
        X = [ epochs[np.where(labels==i)] for i in classes] # Extrair épocas de cada classe
        X = np.r_[X[0],X[1]]
        
        y = np.r_[np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))] # target vector
        start = time() # start timer
        args = X, y, filtro, order, fl, fh, fs, nf, n_comp, n_bands, Clog, folds, test_size
        acc = csp_lda_cross(args)
        cost = time() - start # stop timer (cost estimate)
        print(suj, str(round(acc*100,2))+'%', str(int(cost*1000))+'ms')
    