# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import pickle
import math
import itertools
import numpy as np
import pandas as pd
from time import time
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.linalg import eigh
from hyperopt import base, tpe, fmin, hp
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

suj = 2
class_ids = [1, 2]
path = '/mnt/dados/eeg_data/BCI4_2a/'
opt_path = '/home/vboas/teste_opt'
opt_iter = 1000 

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

def extractEpochs(data, e, smin, smax, ev_id): # (loadedData, ev, smin, smax, [1, 2])
    events_list = e[:, 1] #e[:, 2]
    cond = False
    for i in range(len(ev_id)): cond += (events_list == ev_id[i])
    idx = np.where(cond)[0]
    s = e[idx, 0]
    sBegin = s + smin
    sEnd = s + smax
    n_epochs = len(sBegin)
    n_channels = data.shape[0]
    n_samples = smax - smin
    epochs = np.zeros([n_epochs, n_channels, n_samples])
    labels = events_list[idx]
    bad_epoch_list = []
    for i in range(n_epochs):
        epoch = data[:, sBegin[i]:sEnd[i]]
        # Check if epoch is complete
        if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch
        else:
            print('Incomplete epoch detected...')
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    return epochs, labels

def nanCleaner(data):
    for i in range(data.shape[0]):
        bad_idx = np.isnan(data[i, ...])
        data[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], data[i, ~bad_idx])
    return data

def evaluate(args):
    print(args)
    fl, fh, tmin, tmax, ncomp, ap, filt, clf = args    
    fl, fh, ncomp = int(fl), int(fh), int(ncomp)
    nbands = 1 if ap['option']=='classic' else int(ap['nbands'])
    nbands = (fh-fl)-1 if nbands > (fh-fl) else nbands
    if (tmax-tmin)<1: tmax+=1  # garante janela minima de 1seg
    fs = 250
    nf = 0.5*fs 
    channels = 22
    overlap=True
    smin = math.floor(tmin * fs)
    smax = math.floor(tmax * fs)
    buffer_len = smax - smin
    res_freq = (fs / buffer_len)
    bmin = int(fl / res_freq)
    bmax = int(fh / res_freq)
    test_perc=0.5
    #print(ap['option'], filt['design'], clf['model'])
    #### 01
    data = np.load(path + 'npy/A0' + str(suj) + '_data.npy')[:channels]
    events = np.load(path + 'npy/A0' + str(suj) + '_events.npy')
    epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
    epochs = nanCleaner(epochs)
    test_size = int(len(epochs) * test_perc)
    train_size = int(len(epochs) - test_size)
    train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
    epochsT, labelsT = epochs[:train_size], labels[:train_size]
    epochsV, labelsV = epochs[train_size:], labels[train_size:]
    XT = [epochsT[np.where(labelsT == i)] for i in class_ids]  # Extrair épocas de cada classe
    XV = [epochsV[np.where(labelsV == i)] for i in class_ids]  # Extrair épocas de cada classe

    #### 02
    # for ds in ['T','E']:
    #     data = np.load(path + 'npy_sep/A0' + str(suj) + ds + '_data.npy')[:channels]
    #     events = np.load(path + 'npy_sep/A0' + str(suj) + ds + '_events.npy')
    #     epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
    #     epochs = nanCleaner(epochs)
    #     if ds=='T': XT = [ epochs[np.where(labels==i)] for i in class_ids] # Extrair épocas de cada classe
    #     if ds=='E': XV = [ epochs[np.where(labels==i)] for i in class_ids]
    
    XT = np.concatenate([XT[0],XT[1]]) # Dados treino classes A e B
    XV = np.concatenate([XV[0],XV[1]]) # Dados teste classes A e B
    yT = np.concatenate([class_ids[0] * np.ones(int(len(XT)/2)), class_ids[1] * np.ones(int(len(XT)/2))])
    yV = np.concatenate([class_ids[0] * np.ones(int(len(XV)/2)), class_ids[1] * np.ones(int(len(XV)/2))])

    if clf['model'] == 'LDA':   svc = LDA()
    if clf['model'] == 'Bayes': svc = GaussianNB()
    if clf['model'] == 'SVM':   svc = SVC(kernel=clf['kernel'], C=10 ** (clf['Clog']), probability=True)
    if clf['model'] == 'KNN':   svc = KNeighborsClassifier(n_neighbors=int(clf['neig']), metric='minkowski', p=2)  # minkowski,p=2 -> distancia euclidiana padrão
    if clf['model'] == 'DTree': svc = DecisionTreeClassifier(criterion=clf['crit'], random_state=0) #, max_depth=clf_dict['max_depth'], # min_samples_split=math.ceil(clf_dict['min_samples_split']), # None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
    if clf['model'] == 'MLP':   svc = MLPClassifier(verbose=False, max_iter=10000, tol=0.0001, activation=clf['activ'], 
                                                    solver='adam', learning_rate='constant',
                                                    learning_rate_init=10**clf['eta'], alpha=10**clf['alpha'], 
                                                    hidden_layer_sizes=(int(clf['n_neurons']), int(clf['n_hidden']))) #learning_rate=clf['eta_schedule']['eta_type'], solver=clf['solver']['solver_type'],
    st=time()
    if ap['option'] == 'classic':
        if filt['design'] == 'DFT':
            filtered = fft(XT)
            REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
            IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1)) 
            filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
            XTF = np.transpose(filtered, (1, 2, 0))
            filtered = fft(XV)
            REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
            IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1))
            filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
            XVF = np.transpose(filtered, (1, 2, 0))
        else:
            if fl == 0: fl = 0.001
            low = fl / nf
            high = fh / nf
            if high >= 1: high = 0.99
            if filt['design'] == 'IIR':
                b, a = iirfilter(int(filt['iir_order']), [low, high], btype='bandpass') # butter() or iirfilter()
            elif filt['design'] == 'FIR':
                b = firwin(int(filt['fir_order']), [low, high], window='hamming', pass_zero=False)
                a = [1]   
            XTF = lfilter(b, a, XT) # lfilter() or filtfilt()
            XVF = lfilter(b, a, XV)
        csp = CSP(n_components=ncomp)
        csp.fit(XTF, yT)
        XT_CSP = csp.transform(XTF)
        XV_CSP = csp.transform(XVF)
        svc.fit(XT_CSP, yT)
        scores = svc.predict(XV_CSP)
        acc = np.mean(scores == yV)           
    
    if ap['option'] == 'sbcsp':
        if filt['design'] == 'DFT': 
            filtered = fft(XT)
            REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
            IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1)) 
            filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
            XTD = np.transpose(filtered, (1, 2, 0))
            filtered = fft(XV)
            REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
            IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1))
            filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
            XVD = np.transpose(filtered, (1, 2, 0))
            nbins = len(XTD[0,0,:])
            print(nbins)
        else: nbins = fh-fl 
        step = int(nbins/nbands)
        size = int(step/0.5) if overlap else step 
        SCORE_T = np.zeros((len(XT), nbands))
        SCORE_V = np.zeros((len(XV), nbands))
        for i in range(nbands):
            if filt['design'] == 'DFT':
                bini = i*step
                bend = i*step+size
                if bend >= nbins: bend = nbins-1
                XTF = XTD[:, :, bini:bend]
                XVF = XVD[:, :, bini:bend]
            else:
                fl_sb = i*step+fl
                fh_sb = i*step+size+fl
                if fh_sb > fh: fh_sb = fh
                if fl_sb == 0: fl_sb = 0.001
                low = fl_sb/nf
                high = fh_sb/nf
                if high >= 1: high = 0.99
                if filt['design'] == 'IIR':
                    b, a = iirfilter(int(filt['iir_order']), [low, high], btype='bandpass') # butter() or iirfilter()
                elif filt['design'] == 'FIR':
                    b, a = firwin(int(filt['fir_order']), [low, high], window='hamming', pass_zero=False), [1]  
                XTF = lfilter(b, a, XT) # lfilter() or filtfilt()    
                XVF = lfilter(b, a, XV)
            csp = CSP(n_components=ncomp)
            csp.fit(XTF, yT)
            XT_CSP = csp.transform(XTF) 
            XV_CSP = csp.transform(XVF)
            lda = LDA()
            lda.fit(XT_CSP, yT)
            SCORE_T[:, i] = np.ravel(lda.transform(XT_CSP)) 
            SCORE_V[:, i] = np.ravel(lda.transform(XV_CSP)) 
        # Meta-classificador Bayesiano
        SCORE_T0 = SCORE_T[yT == class_ids[0], :]
        SCORE_T1 = SCORE_T[yT == class_ids[1], :]
        p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0)) 
        p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))   
        META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
        META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
        svc.fit(META_SCORE_T, yT)
        scores = svc.predict(META_SCORE_V)
        acc = np.mean(scores == yV)

    cost=time()-st 
    return acc * (-1)


if __name__ == '__main__':
    
    # print(evaluate((0, 37, 2.0, 4.0, 10, {'option':'sbcsp','nbands':21}, {'design':'DFT','iir_order':5}, 
    #                 {'model':'MLP', 'eta':-2, 'alpha':-2, 'n_neurons':150, 'n_hidden':1, 'activ':'identity'})))
    
    print(evaluate((0, 43, 2.0, 5.5, 20, {'option':'sbcsp','nbands':34}, {'design':'DFT','iir_order':5}, {'model':'SVM', 'kernel':'rbf','Clog':-7})))

    # print(evaluate((10, 30, 3.0, 5.0, 18, {'option':'classic'}, {'design':'DFT','iir_order':5}, {'model':'KNN', 'neig':42})))
    
    space = (   
        hp.quniform('fl', 0, 10, 1),
        hp.quniform('fh', 30, 51, 1),
        hp.quniform('tmin', 2, 4, 0.5),
        hp.quniform('tmax', 4, 6, 0.5),
        hp.quniform('ncomp', 2, 21, 2),
        hp.choice('approach', [
            {'option':'classic',},
            {'option':'sbcsp', 'nbands': hp.quniform('nbands', 2, 50, 1)}]),
        hp.choice('filt', [
            {'design':'DFT',},
            #{'design':'IIR', 'iir_order': hp.quniform('iir_order', 1, 8, 1)},
            #{'design':'FIR', 'fir_order': hp.quniform('fir_order', 1, 8, 1)}
            ]),
        hp.choice('clf', [
            {'model':'Bayes'},
            {'model':'LDA'},
            {'model':'KNN', 'neig': hp.quniform('neig', 2, 50, 1)},
            {'model':'SVM', 'Clog': hp.quniform('Clog', -8, 4, 1), 
              'kernel': hp.choice('kernel', ['linear','poly','sigmoid','rbf'])}, # [{'kf':'linear'},{'kf':'poly'},{'kf':'sigmoid'},{'kf':'rbf','width': hp.lognormal('width', 0, 1)}]
            {'model':'MLP', 'eta': hp.quniform('eta', -8, 1, 1), 'alpha': hp.quniform('alpha', -8, 1, 1), 
              'n_neurons' : hp.quniform('n_neurons', 50, 500, 50), 'n_hidden': hp.quniform('n_hidden', 1, 4, 1),
              'activ': hp.choice('activ', ['identity','logistic','tanh','relu'])}, # [{'af':'identity'},{'af':'logistic'},{'af':'tanh'},{'af':'relu'}] 
            {'model':'DTree', 'crit': hp.choice('crit', ['gini', 'entropy'])}]))
    
          
    
    trials = base.Trials()  
    # try:
    #     print('Trying to pickle file')
    #     trials = pickle.load(open(opt_path, 'rb'))
    # except:
    #     print('No trial file at specified path, creating new one')
    #     trials = base.Trials()
    # else:
    #     print('File found')
    
    try:
        print('Size of object: ' + str(len(trials)))
        best = fmin(evaluate, space=space, algo=tpe.suggest, max_evals=len(trials) + opt_iter, trials=trials)
        pickle.dump(trials, open(opt_path, 'wb'))
    except:
        print('Exception raised')
        pickle.dump(trials, open(opt_path, 'wb'))
        raise
    # print(best)
    
    print('\n### BEST SETUP ###')
    print(f'Subject: {suj}')
    print(f'Classes: {class_ids}')
    print(f"Accuracy: {round(trials.best_trial['result']['loss']*-100, 2)}%")
    print('Approach: {}'.format('Classic' if best['approach']==0 else 'SBCSP'))
    print('Band: {}-{}Hz'.format(int(best['fl']),int(best['fh'])))
    if best['approach']==1: print('N Sub-bands: {}'.format(int(best['nbands'])))
    print('Filtering: {}'.format('DFT' if best['filt']==0 else 'IIR'))
    if best['filt'] in [1,2]: print('order: {}'.format(int(best['iir_order'])))
    print('EEG Segment: {}-{}s'.format(best['tmin'],best['tmax']))
    print('N CSP: {}'.format(int(best['ncomp'])))
    if best['clf'] == 0:    print('Classifier: Bayes')
    elif best['clf'] == 1:  print('Classifier: LDA')
    elif best['clf'] == 2:  print(f"Classifier: KNN {int(best['neig'])}")
    elif best['clf'] == 3: 
        kf = 'linear' if best['kernel']==0 else 'poly' if best['kernel']==1 else 'sigmoid' if best['kernel']==2  else 'rbf'
        print(f"Classifier: SVM k={kf} C=10**({best['Clog']})")
    elif best['clf'] == 4:
        af = 'identity' if best['activ']==0 else 'logistic' if best['activ']==1 else 'tanh' if best['activ']==2  else 'relu'
        print(f"Classifier: MLP ({int(best['n_neurons'])}, {int(best['n_hidden'])}) af={af}  eta={best['eta']} alpha={best['alpha']} ")
    elif best['clf'] == 5:  
        crit = 'gini' if best['crit']==0 else 'entropy'
        print(f"Classifier: DTree {crit}")
    
    print('\n', trials.best_trial['misc']['vals'])