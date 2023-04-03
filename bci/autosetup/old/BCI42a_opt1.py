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
from scipy.signal import lfilter, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

fs = 250
nf = fs/2.

order = 5
nchannels = 22

suj = 1
cl = [1, 2]
path = '/mnt/dados/eeg_data/BCI4_2a/npy_epochs/A0'

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
    #print('csp_lda: ', args, end='')
    filt, fl, fh, tmin, tmax, n_comp, classifier = args
    XT = np.load(path + str(suj) + 'T.npy')
    XV = np.load(path + str(suj) + 'E.npy')
    XT = ([XT[cl[0]-1], XT[cl[1]-1]])
    XV = ([XV[cl[0]-1], XV[cl[1]-1]])
    XT = np.concatenate([XT[0],XT[1]]) # Dados de treinamento das classes A, B
    XV = np.concatenate([XV[0],XV[1]]) # Dados de validação das classes A, B
    y = np.concatenate([np.zeros(int(len(XT)/2)), np.ones(int(len(XT)/2))]) # Vetor gabarito
    smin = int(tmin * fs)
    smax = int(tmax * fs)
    XTJ = XT[:,:,smin:smax] 
    XVJ = XV[:,:,smin:smax]
    start = time() # start timer
    if filt['filt']['filt_type'] == 'FFT': # FFT - Spectral filtering
        bmin = int(fl * (fs/nf))
        bmax = int(fh * (fs/nf))
        filtered = fft(XTJ)
        REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1)) 
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        XTF = np.transpose(filtered, (1, 2, 0))       
        filtered = fft(XVJ)
        REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1))
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
        XVF = np.transpose(filtered, (1, 2, 0))
    elif filt['filt']['filt_type'] == 'IIR': # IIR - Temporal filtering
        if fl==0: fl=0.001
        Wnl = fl/nf
        Wnh = fh/nf
        if Wnh >= 1: Wnh = 0.99
        b, a = butter(filt['filt']['order'], [Wnl, Wnh], btype='bandpass') # to filt IIR
        XTF = lfilter(b, a, XTJ)
        XVF = lfilter(b, a, XVJ)
    csp = CSP(n_components=n_comp)
    csp.fit(XTF, y)
    XT_CSP = csp.transform(XTF)
    XV_CSP = csp.transform(XVF) 
    
    if classifier['clf']['type'] == 'LDA': 
        clf = LinearDiscriminantAnalysis()
    
    if classifier['clf']['type'] == 'SVM': 
        clf = SVC(kernel=classifier['clf']['kernel']['ktype'], 
                  C=10**(classifier['clf']['C']))
    
    if classifier['clf']['type'] == 'KNN': 
        clf = KNeighborsClassifier(n_neighbors=int(classifier['clf']['n_neighbors']), 
                                   metric='minkowski', p=2) #minkowski e p=2 -> para usar distancia euclidiana padrão
    
    if classifier['clf']['type'] == 'DTree': 
        clf = DecisionTreeClassifier(criterion=classifier['clf']['criterion'], 
                                     max_depth=classifier['clf']['max_depth'], 
                                     min_samples_split=math.ceil(classifier['clf']['min_samples_split']), 
                                     random_state=0) #None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
    
    if classifier['clf']['type'] == 'Bayes': 
        clf = GaussianNB()
   
    if classifier['clf']['type'] == 'MLP': 
        clf = MLPClassifier(verbose=False, 
                            max_iter=10000, 
                            tol=0.0001, 
                            activation=classifier['clf']['activation']['act_type'], 
                            learning_rate_init=classifier['clf']['eta'], 
                            learning_rate=classifier['clf']['eta_schedule']['eta_type'],  
                            solver=classifier['clf']['solver']['solver_type'],
                            hidden_layer_sizes=(int(classifier['clf']['hidden_n_neurons']),int(classifier['clf']['n_hidden'])))
    
    clf.fit(XT_CSP, y)
    scores = clf.predict(XV_CSP)
    cost = time() - start # stop timer
    acc = np.mean(scores == y) 
    return acc, cost

def sbcsp(args):
    #print('sbcsp: ', args, end='')
    filt, fl, fh, tmin, tmax, n_comp, n_bands, classifier = args
    XT = np.load(path + str(suj) + 'T.npy')
    XV = np.load(path + str(suj) + 'E.npy')
    XT = ([XT[cl[0]-1], XT[cl[1]-1]])
    XV = ([XV[cl[0]-1], XV[cl[1]-1]])
    XT = np.concatenate([XT[0],XT[1]]) # Dados de treinamento das classes A, B
    XV = np.concatenate([XV[0],XV[1]]) # Dados de validação das classes A, B
    y = np.concatenate([np.zeros(int(len(XT)/2)), np.ones(int(len(XT)/2))]) # Vetor gabarito
    smin = int(tmin * fs)
    smax = int(tmax * fs)
    XTJ = XT[:,:,smin:smax] 
    XVJ = XV[:,:,smin:smax]
    start = time() # start timer
    if filt['filt']['filt_type'] == 'FFT': # FFT - Spectral filtering
        bmin = int(fl * (fs/nf))
        bmax = int(fh * (fs/nf))
        filtered = fft(XTJ)
        REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1)) 
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        XT_FFT = np.transpose(filtered, (1, 2, 0))       
        filtered = fft(XVJ)
        REAL = np.transpose(np.real(filtered)[:,:,bmin:bmax], (2, 0, 1))
        IMAG = np.transpose(np.imag(filtered)[:,:,bmin:bmax], (2, 0, 1))
        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
        XV_FFT = np.transpose(filtered, (1, 2, 0))
        
    # Divide sub-bands
    if filt['filt']['filt_type'] == 'FFT': n_bins = len(XT_FFT[0,0,:]) # ou (fh-fl) * 4 # Número total de bins de frequencia 
    elif filt['filt']['filt_type'] == 'IIR': n_bins = fh-fl 
    overlap = 2
    step = int(n_bins / n_bands)
    size = int(step * overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
    
    # Make sub-bands limits and Temporal/Spectral filtering
    SCORE_T = np.zeros((len(XT), n_bands))
    SCORE_V = np.zeros((len(XV), n_bands))
    for i in range(n_bands):
        if filt['filt']['filt_type'] == 'FFT': # Only FFT Spectral filtering
            bin_ini = i*step
            bin_fim = i*step+size
            if bin_fim >= n_bins: bin_fim = n_bins-1
            XTF = XT_FFT[:, :, bin_ini:bin_fim]
            XVF = XV_FFT[:, :, bin_ini:bin_fim]
        elif filt['filt']['filt_type'] == 'IIR': # Only IIR Temporal filtering
            fl_sb = i*step+fl
            fh_sb = i*step+size+fl
            if fl_sb == 0: fl_sb = 0.001
            if fh_sb > fh: fh_sb = fh
            #print(fl_sb, fh_sb, nf, fl_sb/nf, fh_sb/nf)
            if fl_sb==0: fl_sb=0.001
            Wnl = fl_sb/nf
            Wnh = fh_sb/nf
            if Wnh >= 1: Wnh = 0.99
            
            b, a = butter(filt['filt']['order'], [Wnl, Wnh], btype='bandpass') # to filt IIR
            XTF = lfilter(b, a, XTJ) 
            XVF = lfilter(b, a, XVJ)
        
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
    #svc = SVC(kernel="linear", C=10**Clog)
    
    if classifier['clf']['type'] == 'LDA': 
        svc = LinearDiscriminantAnalysis()
    
    if classifier['clf']['type'] == 'SVM': 
        svc = SVC(kernel=classifier['clf']['kernel']['ktype'], 
                  C=10**(classifier['clf']['C']))
    
    if classifier['clf']['type'] == 'KNN': 
        svc = KNeighborsClassifier(n_neighbors=int(classifier['clf']['n_neighbors']), 
                                   metric='minkowski', p=2) #minkowski e p=2 -> para usar distancia euclidiana padrão
    
    if classifier['clf']['type'] == 'DTree': 
        svc = DecisionTreeClassifier(criterion=classifier['clf']['criterion'], 
                                     max_depth=classifier['clf']['max_depth'], 
                                     min_samples_split=math.ceil(classifier['clf']['min_samples_split']), 
                                     random_state=0) #None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
    
    if classifier['clf']['type'] == 'Bayes': 
        svc = GaussianNB()
   
    if classifier['clf']['type'] == 'MLP': 
        svc = MLPClassifier(verbose=False, 
                            max_iter=10000, 
                            tol=0.0001, 
                            activation=classifier['clf']['activation']['act_type'], 
                            learning_rate_init=classifier['clf']['eta'], 
                            alpha=classifier['clf']['alpha'],
                            learning_rate=classifier['clf']['eta_schedule']['eta_type'],  
                            solver=classifier['clf']['solver']['solver_type'],
                            hidden_layer_sizes=(int(classifier['clf']['hidden_n_neurons']),int(classifier['clf']['n_hidden'])))
    
    svc.fit(META_SCORE_T, y)
    scores = svc.predict(META_SCORE_V)
    
    cost = time() - start # stop timer (cost estimate)
    acc = np.mean(scores == y) 

    return acc, cost

def fit_model(args):
    filt_, fl, fh, tmin, tmax, ncomp, approach, clf_type = args
    #print(args)
    #print(clf_type['clf']['type'])
    #if clf_type['clf']['type'] == 'DTree': print(clf_type['clf']['min_samples_split'])
    
    if (tmax-tmin)<1: tmax=tmax+1 # garante janela minima de 1seg
 
    if approach['approach']['option'] == 'csp-lda':
        args = ( filt_, int(fl), int(fh), tmin, tmax, int(ncomp), clf_type )
        print(args, '\n\n')
        acc, cost = csp_lda(args)
    if approach['approach']['option'] == 'sbcsp':
        args = ( filt_, int(fl), int(fh), tmin, tmax, int(ncomp), int(approach['approach']['n_sbands']), clf_type )   
        print(args, '\n\n')
        acc, cost = sbcsp(args)
    return acc*(-1)

## Espaço de parâmetros para auto ajuste
space = ({'filt' : hp.choice('filtering', [{'filt_type':'FFT',},{'filt_type':'IIR', 'order' : hp.quniform('filt_iir_order', 2, 7, 1)}])},
         hp.quniform('fl', 0, 8, 1), 
         hp.quniform('fh', 30, 51, 1),
         hp.quniform('tmin', 2, 4, 0.5),
         hp.quniform('tmax', 4, 6, 0.5),
         hp.quniform('ncomp', 2, nchannels, 2),
         
         {'approach' : hp.choice('app_test', [{'option':'csp-lda',}, {'option':'sbcsp', 'n_sbands' : hp.quniform('sbcsp_n_sbands', 2, (30-8)-1, 1)},])},
         
         {'clf' : hp.choice('clf_type', 
                  [{'type': 'Bayes',}, 
                   {'type': 'LDA',}, 
                   {'type': 'KNN', 'n_neighbors': hp.quniform('n_neighbors', 2, 50, 1),},
                   
                  {'type': 'SVM', 
                   'C': hp.quniform('Clog', -8, 4, 1), #hp.lognormal('svm_C', 0, 1),
                   'kernel': hp.choice('svm_kernel', [ {'ktype': 'linear'}, {'ktype':'poly'}, {'ktype': 'rbf', 'width': hp.lognormal('svm_rbf_width', 0, 1)}]),},
                  
                  {'type': 'MLP', 
                   # 'eta': hp.quniform('eta', 0.0001, 0.1, 0.0001),
                   # 'eta': hp.choice('eta', [0.1,0.01,0.001,0.0001]),
                   'eta': hp.lognormal('eta', 0, 1),
                   'eta_schedule': hp.choice('eta_schedule', [ {'eta_type': 'constant'}, {'eta_type': 'invscaling'}, {'eta_type': 'adaptive'} ]),
                   'solver': hp.choice('solver', [ {'solver_type': 'lbfgs'}, {'solver_type': 'sgd'}, {'solver_type': 'adam'} ]),
                   'alpha': hp.lognormal('alpha', 0, 1),
                   'hidden_n_neurons' : hp.quniform('hidden_n_neurons', 50, 500, 50),
                   'n_hidden': hp.quniform('n_hidden', 1, 4, 1),
                   'activation': hp.choice('activation', [ {'act_type': 'identity'}, {'act_type': 'logistic'}, {'act_type': 'tanh'}, {'act_type': 'relu'}, ]),},
                  
                  {'type': 'DTree',
                   'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
                   'max_depth': hp.choice('dtree_max_depth', [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
                   'min_samples_split': hp.quniform('dtree_min_samples_split', 2, 100, 1),},])})
                   #'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),},])})
    
opt_path = '/home/vboas/teste_opt'
opt_iter = 10       

# trials = base.Trials()  
try:
    print('Trying to pickle file')
    trials = pickle.load(open(opt_path, 'rb'))
except:
    print('No trial file at specified path, creating new one')
    trials = base.Trials()
else:
    print('File found')

# best = fmin(fit_model, space=space, algo=tpe.suggest, max_evals=len(trials) + opt_iter, trials=trials)
# print(best)
# pickle.dump(trials, open(opt_path, 'wb'))
try:
    print('Size of object: ' + str(len(trials)))
    best = fmin(fit_model, space=space, algo=tpe.suggest, max_evals=len(trials) + opt_iter, trials=trials)
    pickle.dump(trials, open(opt_path, 'wb'))
except:
    print('Exception raised')
    pickle.dump(trials, open(opt_path, 'wb'))
    raise

print('\n### BEST SETUP ###')
# print('Approach: {}'.format('Classic' if best['approach']==1 else 'SBCSP'))
# print('Filtering: {}'.format('FFT' if best['filt_type']==1 else 'IIR'))
print('Approach: {}'.format('Classic' if best['app_test']==0 else 'SBCSP'))
print('Filtering: {}'.format('FFT' if best['filtering']==0 else 'IIR'))
print('Low Frequency: {}'.format(int(best['fl'])))
print('High Frequency: {}'.format(int(best['fh'])))
print('Start Time Buffer: {}'.format(best['tmin']))
print('End Time Buffer: {}'.format(best['tmax']))
if best['filtering']==1: print('Filt IIR Order: {}'.format(int(best['filt_iir_order'])))
print('CSP n components: {}'.format(int(best['ncomp'])))
if best['app_test']==1: print('N sub_bands: {}'.format(int(best['sbcsp_n_sbands'])))
# print('SVM C: log({})'.format(int(best['Clog'])))

# print(best['dtree_min_samples_split'])  
print('\n', trials.best_trial['misc']['vals'])

# CAPTURAR OS CAMPOS EM best informando o último label (direto) em caso de campos aninhados


##################### BKPs ###############################

# def fit_model(args):
#     approach, filt_type, fl, fh, tmin, tmax, ncomp, nbands, Clog = args
#     if (tmax-tmin)<1: tmax=tmax+1 # garante janela minima de 1seg
#     if approach == 1:
#         args = ( int(ncomp), int(filt_type), int(fl), int(fh), tmin, tmax )
#         acc, cost = csp_lda(args)
#     if approach == 2:
#         args = ( int(filt_type), int(fl), int(fh), tmin, tmax, int(ncomp), int(nbands), Clog )   
#         acc, cost = sbcsp(args)
#     return acc*(-1)   


# space = (hp.quniform('approach', 1, 2, 1),
#           hp.quniform('filt_type', 1, 2, 1),
#           hp.quniform('fl', 0, 8, 1), 
#           hp.quniform('fh', 30, 51, 1),
#           hp.quniform('tmin', 2, 4, 0.5), 
#           hp.quniform('tmax', 4, 6, 0.5),
#           hp.quniform('ncomp', 2, nchannels, 2),
#           hp.quniform('nbands', 2, (30-8)-1, 1), 
#           hp.quniform('Clog', -8, 4, 1),         
# hp.choice('clf_type', [{'type': 'naive_bayes'}, {'type': 'svm'}, {'type': 'dtree'}]))

# space = {'approach': hp.quniform('approach', 1, 2, 1),
#           'filt_type': hp.quniform('filt_type', 1, 2, 1),
#           'fl': hp.quniform('fl', 0, 8, 1), 
#           'fh': hp.quniform('fh', 30, 51, 1),
#           'tmin': hp.quniform('tmin', 2, 4, 0.5), 
#           'tmax': hp.quniform('tmax', 4, 6, 0.5),
#           'ncomp': hp.quniform('ncomp', 2, nchannels, 2),
#           'nbands': hp.quniform('nbands', 2, (30-8)-1, 1), 
#           'Clog': hp.quniform('Clog', -8, 4, 1),
#           'clf_type': hp.choice('clf_type', [
#                   {'type': 'naive_bayes',},
                  
#                   {'type': 'svm', 
#                     'C': hp.lognormal('svm_C', 0, 1),
#                     'kernel': hp.choice('svm_kernel', [ {'ktype': 'linear'}, {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)}, ]),},
                  
#                   {'type': 'dtree',
#                     'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
#                     'max_depth': hp.choice('dtree_max_depth', [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
#                     'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),},
#                   ])}
    

