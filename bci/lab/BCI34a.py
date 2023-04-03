# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import math
import itertools
import numpy as np
from time import time
import pandas as pd
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.io import loadmat
from scipy.fftpack import fft
from scipy.linalg import eigh
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from bci_utils import CSP, extractEpochs, nanCleaner, corrigeNaN

""" 5 subjects | 2 classes (RH, FooT) | 118 channels | Fs 100Hz| 140 per class
    Epoch distribution:
        aa : train=168 test=112  
        al : train=224 test=56
        av : train=84  test=196
        aw : train=56  test=224
        ay : train=28  test=252
"""

def load_mat_info(path, suj):
    mat = loadmat(path + 'mat/' + suj + '.mat')
    data = 0.1 * mat['cnt'].T # convert to uV
    pos = mat['mrk'][0][0][0][0]
    #data = corrigeNaN(data)
    true_mat = loadmat(path + 'mat/' + 'true_labels/trues_' + suj + '.mat')
    true_y = np.ravel(true_mat['true_y']) # RH=1 Foot=2
    # true_test_idx = np.ravel(true_mat['test_idx'])
    events = np.c_[pos, true_y]
    return data, events

def csp_lda_cross(args):
    X, y, filtro, order, fl, fh, fs, nf, n_comp, n_bands, Clog, folds = args
    if filtro == 'FFT': # Spectral filtering
        XFF = fft(X)
        bin0 = int(fl * (fs/nf))
        binN = int(fh * (fs/nf))
        REAL = np.transpose(np.real(XFF)[:,:,bin0:binN], (2, 0, 1))
        IMAG = np.transpose(np.imag(XFF)[:,:,bin0:binN], (2, 0, 1)) 
        XF0 = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        XF = np.transpose(XF0, (1, 2, 0))     
    elif filtro == 'IIR': # Temporal filtering
        if fl==0: fl=0.001
        b, a = butter(order, [fl/nf, fh/nf], btype='bandpass')
        XF = lfilter(b, a, X) 
    #process = Pipeline([('CSP', CSP(n_comp)), ('classifier', KNeighborsClassifier(n_neighbors=17, metric='minkowski', p=2))])
    process = Pipeline([('CSP', CSP(n_comp)), ('classifier', LinearDiscriminantAnalysis())])
    #kf = StratifiedShuffleSplit(folds, test_size=test_size, random_state=0)
    kf = StratifiedKFold(folds, False)
    scores = cross_val_score(process, XF, y, cv=kf) ## Cross-validation:
    return np.mean(scores)

def sbcsp_cross(args):
    X, y, filtro, order, fl, fh, fs, nf, n_comp, n_bands, Clog, folds = args
    kf = StratifiedKFold(folds, False)
    cross_scores = []
    for idx_treino, idx_teste in kf.split(X, y):
        XT = X[idx_treino]
        XV = X[idx_teste]
        yT = y[idx_treino]
        yV = y[idx_teste]
        if filtro == 'FFT': # Only Spectral filtering
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
            if filtro == 'FFT': #Spectral filtering
                bin_ini = i*step
                bin_fim = i*step+size
                if bin_fim >= n_bins: bin_fim = n_bins-1
                XTF = XT_FFT[:, :, bin_ini:bin_fim]
                XVF = XV_FFT[:, :, bin_ini:bin_fim]
            elif filtro == 'IIR': # Temporal filtering
                fl_sb = i*step+fl
                fh_sb = i*step+size+fl
                if fl_sb == 0: fl_sb = 0.001
                if fh_sb > fh: fh_sb = fh
                #print(fl_sb, fh_sb, nf, fl_sb/nf, fh_sb/nf)
                Wnl = fl_sb/nf
                Wnh = fh_sb/nf
                if Wnh >= 1: Wnh = 0.99
                b, a = butter(order, [Wnl, Wnh], btype='bandpass') # to filt IIR
                XTF = lfilter(b, a, XT) # comment here
                XVF = lfilter(b, a, XV) # comment here
            csp = CSP(n_components=n_comp)
            csp.fit(XTF, yT)
            XT_CSP = csp.transform(XTF) 
            XV_CSP = csp.transform(XVF)
            clf = LinearDiscriminantAnalysis()
            clf.fit(XT_CSP, yT)
            SCORE_T[:, i] = np.ravel(clf.transform(XT_CSP)) # classificaçoes de cada época nas N sub bandas - auto validação
            SCORE_V[:, i] = np.ravel(clf.transform(XV_CSP)) # validação
        # Meta-classificador Bayesiano
        SCORE_T0 = SCORE_T[yT == 0, :]
        m0 = np.mean(SCORE_T0, axis=0) #media classe A
        std0 = np.std(SCORE_T0, axis=0) #desvio padrão classe A
        SCORE_T1 = SCORE_T[yT == 1, :]
        m1 = np.mean(SCORE_T1, axis=0)
        std1 = np.std(SCORE_T1, axis=0)
        p0 = norm(m0, std0) # p0 e p1 representam uma distribuição normal de médias m0 e m1, e desvio padrão std0 e std1
        p1 = norm(m1, std1)
        META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
        META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
        # SVM on top of the meta-classifier
        svc = SVC(kernel="linear", C=10**Clog)
        svc.fit(META_SCORE_T, yT)
        scores = svc.predict(META_SCORE_V)
        acc = np.mean(scores == yV)
        cross_scores.append(acc)
    return np.mean(cross_scores)

if __name__ == '__main__':
    path = '/mnt/dados/eeg_data/BCI3_4a/'
    subjects = ['aa','al','av','aw','ay']
    classes = [1,2] # 1:rh 2:foot
    channels = 118
    fs = 100
    nf = fs/2
    Tmin, Tmax = 0.5, 2.5 # Start trial= 0; Start cue=0; Start MI= 0; End trial(break)= 5
    smin = int(math.floor(Tmin * fs)) # initial sample (ex. 0)
    smax = int(math.floor(Tmax * fs)) # final sample (ex. 2500)
    filtro = 'IIR' #'IIR' or 'FFT'
    order = 5
    fl = 8
    fh = 30
    n_comp = 13
    n_bands = 8
    Clog = -3
    folds = 10
    test_perc = 0.2

    # for suj in subjects: # only for convert mat to npy files
    #     data, events = load_mat_info(path, suj)
    #     data1 = np.asarray([ np.nan_to_num(dt) for dt in data ])
    #     data = np.asarray([ np.ravel(pd.DataFrame(dt).fillna(pd.DataFrame(dt).mean())) for dt in data ])        
    #     np.save(path + 'npy_all/' + suj + '_events', events)  
    #     np.save(path + 'npy_all/' + suj + '_data', data)
    
    RES = []
    for suj in subjects:
        events = np.load(path + 'npy_all/' + suj + '_events.npy')
        data = np.load(path + 'npy_all/' + suj + '_data.npy')[:channels]
        epochs, labels = extractEpochs(data, events, smin, smax, classes)
        epochs = np.asarray([ nanCleaner(ep) for ep in epochs ])
        X = [ epochs[np.where(labels==i)] for i in classes]
        X = np.concatenate([X[0],X[1]]) # X[0]=RH, X[1]=FT
        y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) # vetor gabarito
        start = time() # start timer
        args = (X, y, filtro, order, fl, fh, fs, nf, n_comp, n_bands, Clog, folds)
        acc = csp_lda_cross(args)
        #acc = sbcsp_cross(args)
        cost = time() - start # stop timer (cost estimate)
        RES.append([suj,acc,cost])
        print(suj, str(round(acc*100,2))+'%', str(round(cost,1))+'s') 
    RES = np.array(RES)
    print(f'{str(round(np.asarray(RES[:,1], dtype=float).mean()*100,2))}%  +- {str(round(np.asarray(RES[:,1], dtype=float).std()*100,1))}% || {round(np.asarray(RES[:,2], dtype=float).sum(),1)}s')
    
    
    