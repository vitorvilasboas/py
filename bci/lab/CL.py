# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
""" 1 subject (CL) | 3 classes (lh, rh, foot) | 16 channels | Fs 125Hz
    lh-rh -> 100 trials (50 per class) 5*20 - 1 session
    lh-ft -> 48 trials (24 per class) 3*16 - 1 session
"""
import math
import itertools
import numpy as np
from time import time
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.linalg import eigh
from scipy.signal import lfilter, butter
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from bci_utils import CSP

def csp_lda(args):
    XT, XV, yT, yV, filtro, order, fl, fh, fs, nf, res_freq, n_comp, n_bands, Clog = args
    if filtro == 'DFT': # Spectral filtering
        filtered = fft(XT)
        bmin = int(fl / res_freq)
        bmax = int(fh / res_freq)
        REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1)) 
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        XTF = np.transpose(filtered, (1, 2, 0))       
        filtered = fft(XV)
        REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1))
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
        XVF = np.transpose(filtered, (1, 2, 0))
    elif filtro == 'IIR': # Temporal filtering
        if fl==0: fl=0.001
        b, a = butter(order, [fl/nf, fh/nf], btype='bandpass') # to filt IIR
        XTF = lfilter(b, a, XT)
        XVF = lfilter(b, a, XV)
    csp = CSP(n_components=n_comp)
    csp.fit(XTF, yT)
    XT_CSP = csp.transform(XTF)
    XV_CSP = csp.transform(XVF) 
    clf = LinearDiscriminantAnalysis()
    #clf = SVC(kernel="poly", C=10**(-4))
    #clf = KNeighborsClassifier(n_neighbors=24, metric='minkowski', p=2) #minkowski e p=2 -> para usar distancia euclidiana padrão
    #clf = DecisionTreeClassifier(criterion='entropy', random_state=0) #max_depth = None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
    #clf = GaussianNB()
    #clf = MLPClassifier(verbose=False, max_iter=10000, tol=0.0001, activation='logistic', learning_rate_init=0.001, learning_rate='invscaling',  solver='adam') #hidden_layer_sizes=(100,),
    clf.fit(XT_CSP, yT)
    scores = clf.predict(XV_CSP)
    acc = np.mean(scores == yV) 
    return acc

def csp_lda_cross(args):
    X, y, filtro, order, fl, fh, fs, nf, res_freq, n_comp, n_bands, Clog, folds, test_size = args
    if filtro == 'DFT': # Spectral filtering
        filtered = fft(X)
        bmin = int(fl / res_freq)
        bmax = int(fh / res_freq)
        REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1)) 
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
    ## Cross-validation: 
    process = Pipeline([('CSP', CSP(n_components=n_comp)), ('classifier', clf)]) # executa uma sequencia de processamento com um classificador no final
    cv = StratifiedShuffleSplit(folds, test_size=test_size, random_state=42)
    scores = cross_val_score(process, XF, y, cv=cv)
    return np.mean(scores)

def sbcsp(args):
    XT, XV, yT, yV, filtro, order, fl, fh, fs, nf, res_freq, n_comp, n_bands, Clog
    if filtro == 'DFT':
        filtered = fft(XT)
        bmin = int(fl / res_freq)
        bmax = int(fh / res_freq)
        REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1)) 
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        XT_FFT = np.transpose(filtered, (1, 2, 0))       
        filtered = fft(XV)
        REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1))
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
        XV_FFT = np.transpose(filtered, (1, 2, 0))
        n_bins = len(XT_FFT[0,0,:])
    elif filtro == 'IIR': n_bins = fh-fl 
    overlap = 0.5
    step = int(n_bins / n_bands)
    size = int(step / overlap)
    SCORE_T = np.zeros((len(XT), n_bands))
    SCORE_V = np.zeros((len(XV), n_bands))
    for i in range(n_bands):
        if filtro == 'DFT':
            bin_ini = i*step
            bin_fim = i*step+size
            if bin_fim >= n_bins: bin_fim = n_bins-1
            XTF = XT_FFT[:, :, bin_ini:bin_fim]
            XVF = XV_FFT[:, :, bin_ini:bin_fim]
        elif filtro == 'IIR':
            fl_sb = i*step+fl
            fh_sb = i*step+size+fl
            if fl_sb == 0: fl_sb = 0.001
            if fh_sb > fh: fh_sb = fh 
            Wnl = fl_sb/nf
            Wnh = fh_sb/nf
            if Wnh >= 1: Wnh = 0.99
            b, a = butter(order, [Wnl, Wnh], btype='bandpass')
            XTF = lfilter(b, a, XT) 
            XVF = lfilter(b, a, XV)
        csp = CSP(n_components=n_comp)
        csp.fit(XTF, yT)
        XT_CSP = csp.transform(XTF) 
        XV_CSP = csp.transform(XVF)
        clf = LinearDiscriminantAnalysis()
        clf.fit(XT_CSP, yT)
        SCORE_T[:, i] = np.ravel(clf.transform(XT_CSP))
        SCORE_V[:, i] = np.ravel(clf.transform(XV_CSP))
    SCORE_T0 = SCORE_T[yT == np.unique(yT)[0], :]
    SCORE_T1 = SCORE_T[yT == np.unique(yT)[1], :]
    p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
    p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
    META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
    META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
    svc = SVC(kernel="linear", C=10**Clog)
    svc.fit(META_SCORE_T, yT)
    scores = svc.predict(META_SCORE_V)
    return np.mean(scores == yV) 
       
if __name__ == "__main__":
    path = '/mnt/dados/eeg_data/BCI_LINCE/'
    # channels = 16
    # fs = 125.0
    # nf = fs/2.
    # filtro = 'IIR' #'IIR' or 'DFT'
    # order = 5
    # fl = 4
    # fh = 40
    # n_comp = 2
    # n_bands = 10
    # Clog = -4
    # Tmin, Tmax = 2, 6 # Start trial= 0; Beep= 1; Wait= 2; Start cue= 2; Start MI= 3; End MI= 9; End trial(break)= 14
    # smin = int(math.floor(Tmin * fs)) # amostra inicial (ex. 0)
    # smax = int(math.floor(Tmax * fs)) # amostra final (ex. 2500)
    # buffer_len = smax - smin
    # res_freq = (fs / buffer_len)
    # test_perc = 0.5
    # crossval = False

    for cl in ['LR', 'LF']: #
        data = np.load(path + 'CL/orig_CL_' + cl + '_data.npy').T
        events = np.load(path + 'CL/orig_CL_' + cl + '_events.npy').astype(int)
        # data = corrigeNaN(data)
        for i in range(len(events)-1): 
            if events[i,1]==0: 
                events[i,1] = events[i+1,1]
                events[i+1,1] = events[i+1,1] + 768
        if cl=='LF': events[:,1] = np.where(events[:,1] == 2, 3, events[:,1])
        print(events[:,1])
        # np.save(path + 'CL/CL_' + cl + '_data.npy', data)
        # np.save(path + 'CL/CL_' + cl + '_events.npy', events)
        # class_ids = [1, 2]
        # epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
        # epochs = nanCleaner(epochs) # Correção de NaN nas épocas
        # start = time()
        # if crossval:
        #     args = epochs, labels, filtro, order, fl, fh, fs, nf, res_freq, n_comp, n_bands, Clog, 10, 0.2
        #     acc = csp_lda_cross(args)
        # else:
        #     test_size = int(len(epochs) * test_perc)
        #     train_size = int(len(epochs) - test_size)
        #     train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
        #     epochsT, labelsT = epochs[:train_size], labels[:train_size]
        #     epochsV, labelsV = epochs[train_size:], labels[train_size:]
        #     XT = [epochsT[np.where(labelsT == i)] for i in class_ids]  # Extrair épocas de cada classe
        #     XV = [epochsV[np.where(labelsV == i)] for i in class_ids]  # Extrair épocas de cada classe
        #     XT = np.concatenate([XT[0],XT[1]]) # Dados treino classes A e B
        #     XV = np.concatenate([XV[0],XV[1]]) # Dados teste classes A e B
        #     yT = np.concatenate([class_ids[0] * np.ones(int(len(XT)/2)), class_ids[1] * np.ones(int(len(XT)/2))])
        #     yV = np.concatenate([class_ids[0] * np.ones(int(len(XV)/2)), class_ids[1] * np.ones(int(len(XV)/2))])
        #     args = XT, XV, yT, yV, filtro, order, fl, fh, fs, nf, res_freq, n_comp, n_bands, Clog
        #     #acc = csp_lda(args)
        #     acc = sbcsp(args)
        # cost = time() - start
        # print(cl, str(round(acc*100,2))+'%', str(int(cost*1000))+'ms')
        