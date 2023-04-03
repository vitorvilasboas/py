# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 16:25:04 2020
@author: vboas
"""
import os
import mne
import pickle
import warnings
import itertools
import numpy as np
import pandas as pd
from time import time
import seaborn as sns
from sklearn.svm import SVC
from scipy.io import loadmat
from scipy.stats import norm
from functools import partial
from scipy.linalg import eigh
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from hyperopt import base, fmin, tpe, hp
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter, decimate, welch
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, StratifiedKFold
from hyperopt.plotting import main_plot_history, main_plot_histogram, main_plot_vars, main_plot_1D_attachment
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, make_scorer, precision_recall_fscore_support, confusion_matrix

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

def extractEpochs(data, events, smin, smax, class_ids):
    events_list = events[:, 1] # get class labels column
    cond = False
    for i in range(len(class_ids)): cond += (events_list == class_ids[i]) #get only class_ids pos in events_list
    idx = np.where(cond)[0]
    s0 = events[idx, 0] # get initial timestamps of each class epochs
    sBegin = s0 + smin
    sEnd = s0 + smax
    n_epochs = len(sBegin)
    n_channels = data.shape[0]
    n_samples = smax - smin
    epochs = np.zeros([n_epochs, n_channels, n_samples])
    labels = events_list[idx]
    bad_epoch_list = []
    for i in range(n_epochs):
        epoch = data[:, sBegin[i]:sEnd[i]]
        if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch # Check if epoch is complete
        else:
            print('Incomplete epoch detected...', n_samples, '!=', epoch.shape[1])
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    return epochs, labels
    
    
def nanCleaner(epoch):
    """Removes NaN from data by interpolation
    data_in : input data - np matrix channels x samples
    data_out : clean dataset with no NaN samples"""
    for i in range(epoch.shape[0]):
        bad_idx = np.isnan(epoch[i, :])
        epoch[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoch[i, ~bad_idx])
    return epoch
    
    
def corrigeNaN(data):
    for ch in range(data.shape[0] - 1):
        this_chan = data[ch]
        data[ch] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(data[ch])
        meanChannel = np.nanmean(data[ch])
        data[ch, mask] = meanChannel
    return data


class Filter:
    def __init__(self, fl, fh, Fs, filtering, band_type='bandpass'):
        self.ftype = filtering['design']
        if self.ftype != 'DFT':
            nyq = 0.5*Fs
            low = fl/nyq
            high = fh/nyq        
            if low == 0: low = 0.001
            if high >= 1: high = 0.99
            if self.ftype == 'IIR':
                # self.b, self.a = iirfilter(filtering['iir_order'], [low, high], btype='band')
                self.b, self.a = butter(filtering['iir_order'], [low, high], btype=band_type)
            elif self.ftype == 'FIR':
                self.b, self.a = firwin(filtering['fir_order'], [low, high], window='hamming', pass_zero=False), [1]

    def apply_filter(self, X, is_epoch=False):
        if self.ftype != 'DFT': XF = lfilter(self.b, self.a, X) # lfilter, filtfilt
        else:
            XF = fft(X)
            if is_epoch:
                real, imag = np.real(XF).T, np.imag(XF).T
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))))
            else:
                real = np.transpose(np.real(XF), (2, 0, 1))
                imag = np.transpose(np.imag(XF), (2, 0, 1))
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))), (1, 2, 0)) 
        return XF


class CSP:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None
    def fit(self, X, t):
        ch = X.shape[1]
        class_ids = np.unique(t)
        X1 = X[class_ids[0] == t]
        X2 = X[class_ids[1] == t]
        S1, S2 = np.zeros((ch, ch)), np.zeros((ch, ch))
        for i in range(len(X1)): S1 += np.dot(X1[i], X1[i].T) / X1[i].shape[-1] # cov X[i]
        for i in range(len(X2)): S2 += np.dot(X2[i], X2[i].T) / X2[i].shape[-1] # ...sum((X*X.T)/q)
        S1 /= len(X1);
        S2 /= len(X2)
        [D, W] = eigh(S1, S1 + S2) # + 1e-10 * np.eye(22))
        ind = np.empty(ch, dtype=int)
        ind[0::2] = np.arange(ch-1, ch//2 - 1, -1)
        ind[1::2] = np.arange(0, ch//2)
        # W += 1e-1 * np.eye(22)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        return self # used on cross-validation pipeline
    def transform(self, X):
        Y = np.asarray([np.dot(self.filters_, ep) for ep in X])
        # FEAT = np.log(np.mean(Y**2, axis=2))
        FEAT = np.log(np.var(Y, axis=2))
        return FEAT


def design_clf(clf_details):
    if clf_details['model'] == 'LDA': clf = LDA()
    if clf_details['model'] == 'LR':  clf = LogisticRegression(verbose=False)
    if clf_details['model'] == 'SVM':
        clf = SVC(kernel=clf_details['kernel']['kf'], C=10**(clf_details['C']), probability=True)
    if clf_details['model'] == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=int(clf_details['neig'])) #, metric=clf_details['metric'], p=3)
    if clf_details['model'] == 'MLP':
        clf = MLPClassifier(verbose=False, max_iter=10000, learning_rate_init=10 ** clf_details['eta'], activation='tanh', # clf_details['activ']['af'], 
                                 hidden_layer_sizes=(int(clf_details['n_neurons']), int(clf_details['n_hidden'])))
                                 # learning_rate='constant', solver=clf_details['mlp_solver']) # alpha=10**clf_details['alpha'], learning_rate=clf_details['eta_type'],
        clf.out_activation = 'softmax'                    
    if clf_details['model'] == 'Bayes': clf = GaussianNB()
    if clf_details['model'] == 'DTree': clf = DecisionTreeClassifier(criterion=clf_details['crit'], random_state=0)
    return clf
    

def objective_func(args, lb_utils, data1, data2, events1, events2, path_eeg, class_ids=[1,2], Fs=250, filtering={'design':'DFT'},
                   crossval=True, nfolds=5, only_train=False):
    # print(args)
    fl, fh, tmin, tmax, ncsp, nbands, clf_details = args
    
    clf = design_clf(clf_details)
    args = (class_ids, int(fl), int(fh), Fs, int(ncsp), int(nbands), filtering, clf)
    
    while(tmax-tmin) < 1: tmax += 0.5
    smin, smax = int(tmin*Fs), int(tmax*Fs)
    
    # lb_utils = [7,8,9,10]
    # lb_utils = [5,6,7,8] # sujeito 4
    
    ZT, tt = extractEpochs(data1, events1, smin, smax, lb_utils)
    for i,k in zip(lb_utils, range(1, 5)): tt = np.where(tt == i, k, tt)
    ZT = np.vstack([ ZT[np.where(tt == k)] for k in class_ids ])
    tt = np.hstack([ np.ones(len(ZT)//2)*k for k in class_ids ]).astype(int) 
    
    ZV, _ = extractEpochs(data2, events2, smin, smax, [7])
    tv = np.ravel(loadmat(path_eeg + 'true_labels/A0' + str(suj) + 'E.mat')['classlabel'])
    ZV = np.vstack([ ZV[np.where(tv == k)] for k in class_ids ])
    tv = np.hstack([ np.ones(len(ZV)//2)*k for k in class_ids ]).astype(int) 
    
    ZT, ZV = nanCleaner(ZT), nanCleaner(ZV)
    
    # print(np.asarray(ZT).shape)
    # print(np.asarray(ZV).shape)
    
    # print(crossval)
    
    if crossval:
        kf = StratifiedShuffleSplit(nfolds, test_size=1/nfolds, random_state=None)  
        # kf = StratifiedKFold(nfolds, False, random_state=None)
        if only_train: Z, t = np.copy(ZT), np.copy(tt)
        else: Z, t = np.vstack([ZT, ZV]), np.hstack([tt, tv])
        
        cross_acc, cross_kpa = [], []
        for train, test in kf.split(Z, t):
            ZT, tt, ZV, tv = Z[train], t[train], Z[test], t[test]
            acc_fold, kpa_fold = sbcsp_chain(ZT, ZV, tt, tv, args) if (nbands > 1) else classic_chain(ZT, ZV, tt, tv, args)
            cross_acc.append(acc_fold)
            cross_kpa.append(kpa_fold)
            # print(np.asarray(ZT).shape)
            # print(np.asarray(ZV).shape)
        acc, kpa = np.mean(cross_acc), np.mean(cross_kpa)
    
    else: acc, kpa = sbcsp_chain(ZT, ZV, tt, tv, args) if (nbands > 1) else classic_chain(ZT, ZV, tt, tv, args)

    return acc * (-1)


def classic_chain(ZT, ZV, tt, tv, args):
    # print('BU')
    class_ids, fl, fh, Fs, ncsp, nbands, filtering, clf = args
    filt = Filter(fl, fh, Fs, filtering)
    XT = filt.apply_filter(ZT)
    XV = filt.apply_filter(ZV)
    if filtering['design'] == 'DFT':  # extrai somente os bins referentes à banda de interesse
        bsize = 2/(Fs/ZT.shape[-1])  # 2==sen/cos do comp complexo da fft intercalados / resol freq 
        XT = XT[:, :, round(fl * bsize):round(fh * bsize)]
        bsize = 2/(Fs/ZV.shape[-1])  
        XV = XV[:, :, round(fl * bsize):round(fh * bsize)]
    
    # TRAINING 
    csp = mne.decoding.CSP(n_components=ncsp)  # mne.decoding.CSP(n_components=ncsp) || CSP(n_components=ncsp)
    csp.fit(XT, tt)
    csp_filters = csp.filters_
    FT = csp.transform(XT)
    clf.fit(FT, tt)
    
    # EVALUATE
    FV = csp.transform(XV)
    y, yp = clf.predict(FV), clf.predict_proba(FV)
    acc, kpa = np.mean(y == tv), cohen_kappa_score(y, tv)
    
    return acc, kpa


def sbcsp_chain(ZT, ZV, tt, tv, args):
    # print('SBCSP')
    class_ids, fl, fh, Fs, ncsp, nbands, filtering, clf = args
    
    if nbands > (fh-fl): nbands = int(fh-fl)
    step = (fh-fl) / (nbands+1) # n_bins/nbands+1
    size = step / 0.5 # step/overlap
    sub_bands = []
    for i in range(nbands):
        fl_sb = i * step + fl
        fh_sb = i * step + size + fl
        sub_bands.append([fl_sb, fh_sb])
    
    XT, XV = [], []
    if filtering['design'] == 'DFT':
        filt = Filter(fl, fh, Fs, filtering)
        XTF = filt.apply_filter(ZT)
        XVF = filt.apply_filter(ZV)
        for i in range(nbands):
            bsize = 2/(Fs/ZT.shape[-1]) # 2==sen/cos complexo fft intercalados / resol freq
            XT.append(XTF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])
            bsize = 2/(Fs/ZV.shape[-1])
            XV.append(XVF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])         
    elif filtering['design'] == 'IIR':
        for i in range(nbands):
            filt = Filter(sub_bands[i][0], sub_bands[i][1], Fs, filtering)
            XT.append(filt.apply_filter(ZT))
            XV.append(filt.apply_filter(ZV))
    
    # TRAINING        
    csp = [ mne.decoding.CSP(n_components=ncsp) for i in range(nbands) ] # mne.decoding.CSP()
    for i in range(nbands): csp[i].fit(XT[i], tt)
    csp_filters = [csp[i].filters_[:int(ncsp)] for i in range(nbands)]

    FT = [ csp[i].transform(XT[i]) for i in range(nbands) ]
    ldas = [ LDA() for i in range(nbands) ]
    for i in range(nbands): ldas[i].fit(FT[i], tt)
    ST = np.asarray([ np.ravel(ldas[i].transform(FT[i])) for i in range(nbands)]).T # Score LDA
    
    p0 = norm(np.mean(ST[tt == class_ids[0], :], axis=0), np.std(ST[tt == class_ids[0], :], axis=0))
    p1 = norm(np.mean(ST[tt == class_ids[1], :], axis=0), np.std(ST[tt == class_ids[1], :], axis=0))
    META_ST = np.log(p0.pdf(ST) / p1.pdf(ST))
    clf.fit(META_ST, tt)
    
    # EVALUATE
    FV = [ csp[i].transform(XV[i]) for i in range(nbands) ]
    SV = np.asarray([ np.ravel(ldas[i].transform(FV[i])) for i in range(nbands)]).T
    META_SV = np.log(p0.pdf(SV) / p1.pdf(SV))
    y, yp = clf.predict(META_SV), clf.predict_proba(META_SV)
    acc, kpa = np.mean(y == tv), cohen_kappa_score(y, tv) # svm.score(META_SV, tv)
    
    
    # svm = SVC(kernel='linear', C=1e-4, probability=True)
    # svm.fit(META_ST, tt)
    # y, yp = svm.predict(META_SV), svm.predict_proba(META_SV); 
    # acc_svm = round(np.mean(y == tv)*100,2) # round(svm.score(META_SV, tv)*100,2)
    # print('SVM acc:', acc_svm)
    
    # lda = LDA()
    # lda.fit(META_ST, tt)
    # y, yp = lda.predict(META_SV), lda.predict_proba(META_SV)
    # acc_lda = round(np.mean(y == tv)*100,2) # round(lda.score(META_SV, tv)*100,2)
    # print('LDA acc:', acc_lda)
    
    # TT, TV = np.zeros((len(ZT), 2)), np.zeros((len(ZV), 2))
    # for i in range(2): TT[:,i] = np.where(tt == i+1, 1, TT[:,i])
    # for i in range(2): TV[:,i] = np.where(tv == i+1, 1, TV[:,i])
    
    # mlp = MLPClassifier(hidden_layer_sizes=(100,2), max_iter=10000, activation='tanh', verbose=False) #, random_state=42)
    # mlp.out_activation = 'softmax' # 'logistic', 'softmax', # mlp.outputs = 3
    # mlp.fit(META_ST, TT)
    # Y, YP = mlp.predict(META_SV), mlp.predict_proba(META_SV)
    # y = np.argmax(YP, axis=1)+1
    # acc_mlp = round(np.mean(y == tv)*100,2) # round(mlp.score(META_SV, TV)*100,2)
    # print('MLP acc:', acc_mlp)
    
    return acc, kpa


if __name__ == "__main__":
    A = pd.DataFrame(columns=['suj', 'classes', 'exec', 'acc', 'setup'])
    for class_ids in [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]: # [1,2],[1,3],[1,4],[2,3],[2,4],[3,4]
        for suj in range(1, 10):
            for ex in range(1, 11):
                print(suj, class_ids, ex)
                ch=range(0,22) 
                Fs=250
                path_eeg = 'D:/eeg_data/IV2a/'
                path_AS_trials = 'G:/Meu Drive/devs/BCI/results/trials/R4/' + 'A0' + str(suj) + '_ex' + str(ex) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
                filtering = {'design':'DFT'}
                # filtering = {'design':'IIR', 'iir_order':5}
                
                eeg = mne.io.read_raw_gdf(path_eeg + 'A0' + str(suj) + 'T.gdf').load_data()
                data1 = eeg.get_data()[ch] # [channels x samples]
                events1 = mne.events_from_annotations(eeg) # raw.find_edf_events()
                # ch_names = eeg.ch_names
                
                eeg = mne.io.read_raw_gdf(path_eeg + 'A0' + str(suj) + 'E.gdf').load_data()
                data2 = eeg.get_data()[ch] # [channels x samples]
                events2 = mne.events_from_annotations(eeg) # raw.find_edf_events()
                # for k,v in events1[1].items(): print(f'{k}:: {v}')
                # for k,v in events2[1].items(): print(f'{k}:: {v}')
                events1 = np.delete(events1[0], 1, axis=1)
                events2 = np.delete(events2[0], 1, axis=1)
                
                lb_utils = [5,6,7,8] if suj == 4 else [7,8,9,10]
                
                # =============================================================================
                # AUTO SETUP    
                # =============================================================================
                niter = 2
                space = (
                    hp.uniformint('fl', 0, 12),
                    hp.uniformint('fh', 16, 45),
                    hp.quniform('tmin', 0, 2, 0.2),
                    hp.quniform('tmax', 2, 4, 0.2),
                    hp.choice('ncsp', [2, 4, 6, 8, 10, len(ch)]),
                    hp.uniformint('nbands', 1, 22),
                    hp.choice('clf', [
                        {'model':'LR'},
                        {'model':'LDA'},
                        {'model':'KNN', 'neig':hp.uniformint('neig', 5, 20)}, #'metric':hp.choice('metric', ['euclidean', 'manhattan', 'minkowski', 'chebyshev'])},
                        {'model':'SVM', 'C':hp.quniform('C', -8, 0, 4), 'kernel':hp.choice('kernel', [{'kf':'linear'}, {'kf':'poly'}, {'kf':'sigmoid'}, {'kf':'rbf'}])},
                        # {'model':'MLP', 'eta':hp.quniform('eta', -5, -2, 1), 'n_neurons':hp.quniform('n_neurons', 20, 100, 20),  
                        #  'n_hidden':hp.uniformint('n_hidden', 1, 2)}, # 'activ':hp.choice('activ', [{'af':'logistic'}, {'af':'tanh'}])}
                        ]))
                
                trials = base.Trials()
                try: trials = pickle.load(open(path_AS_trials, 'rb'))
                except: trials = base.Trials()
                try:
                    # print('Past trials: ' + str(len(trials)))
                    # init_vals = [{'fl':4, 'fh':40, 'tmin':0.5, 'tmax':2.5, 'ncsp':8, 'nbands':9, 'clf':{'model':'SVM', 'C':-4, 'kernel':{'kf':'linear'}}}]
                    init_vals = [{'fl':8, 'fh':30, 'tmin':0.5, 'tmax':2.5, 'ncsp':8, 'nbands':1, 'clf':{'model':'LDA'}}]
                    obfunc = partial(objective_func, lb_utils=lb_utils, data1=data1, data2=data2, events1=events1, events2=events2, path_eeg=path_eeg, 
                                      class_ids=class_ids, Fs=Fs, filtering=filtering, crossval=True, nfolds=5, only_train=True)
                    best = fmin(obfunc, space=space, algo=tpe.suggest, max_evals=len(trials)+niter, timeout=None, loss_threshold=None, trials=trials, points_to_evaluate=init_vals)
                    pickle.dump(trials, open(path_AS_trials, 'wb'))
                except:
                    pickle.dump(trials, open(path_AS_trials, 'wb')) 
                    raise
                
                # trials = pickle.load(open(path_AS_trials, 'rb'))
                # # best = trials.best_trial['misc']['vals']
                # best = trials.argmin
                
                if best['clf'] == 0: clf = {'model':'LR'}
                elif best['clf'] == 1: clf = {'model':'LDA'}
                elif best['clf'] == 2: clf = {'model':'KNN', 'neig':best['neig']}
                    # metric = 'euclidean' if best['metric']==0 else 'manhattan' if best['metric']==1 else 'minkowski' if best['metric']==2 else 'chebyshev' 
                    # clf = {'model':'KNN', 'metric':metric, 'neig':best['neig']}
                elif best['clf'] == 3:
                    kernel = 'linear' if best['kernel']==0 else 'poly' if best['kernel']==1 else 'sigmoid' if best['kernel']==2 else 'rbf'
                    clf = {'model':'SVM', 'kernel':{'kf':kernel}, 'C':best['C']}
                # elif best['clf'] == 4: clf = {'model':'MLP', 'eta':best['eta'], 'n_neurons':best['n_neurons'], 'n_hidden':best['n_hidden']}
                    # activ = 'logistic' if best['activ']==0 else 'tanh'
                    # clf = {'model':'MLP', 'eta':best['eta'], 'activ':{'af':activ}, 'n_neurons':best['n_neurons'], 'n_hidden':best['n_hidden']}
                ncsp = [2,4,6,8,10,len(ch)][best['ncsp']]
                
                args = (int(best['fl']), int(best['fh']), best['tmin'], best['tmax'], ncsp, int(best['nbands']), clf)
                acc = objective_func(args, lb_utils, data1, data2, events1, events2, path_eeg, class_ids=class_ids, Fs=Fs, filtering=filtering, crossval=False)
                # print(f"Setup: {args}")
                # print('AS   Acc:', round(acc*100,2)*(-1))
                A.loc[len(A)] = [suj, str(class_ids[0]) + ' ' + str(class_ids[1]), ex, acc*(-1), args]

                # # =============================================================================
                # # CMBU   
                # # =============================================================================
                # # t0 = time()
                # # filtering = {'design':'IIR', 'iir_order':5}
                # args1 = (8, 30, 0.5, 2.5, 8, 1, {'model':'LDA'})
                # acc1 = objective_func(args1, lb_utils, data1, data2, events1, events2, path_eeg, class_ids=class_ids, Fs=Fs, filtering=filtering, crossval=False)
                # print('CMBU Acc:', round(acc1*100,2)*(-1))
                # A.loc[len(A)] = [suj, str(class_ids[0]) + ' ' + str(class_ids[1]), None, round(acc1*100,2)*(-1), None]
                # # print('runtime:', round(time()-t0,3), '\n')
                
                # # =============================================================================
                # # CMSB    
                # # =============================================================================
                # # filtering = {'design':'IIR', 'iir_order':5}
                # args2 = (4, 40, 0.5, 2.5, 8, 9, {'model':'SVM', 'C':-4, 'kernel':{'kf':'linear'}})
                # acc2 = objective_func(args2, lb_utils, data1, data2, events1, events2, path_eeg, class_ids=class_ids, Fs=Fs, filtering=filtering, crossval=False)
                # print('CMSB Acc:', round(acc2*100,2)*(-1))
                # A.loc[len(A)] = [suj, str(class_ids[0]) + ' ' + str(class_ids[1]), None, round(acc2*100,2)*(-1), None]
                
                # # =============================================================================
                # # Analysis
                # # =============================================================================
                # trials = pickle.load(open(path_AS_trials, 'rb'))
                # all_loss = [ trials.trials[i]['result']['loss'] * (-1) for i in range(len(trials.trials)) ]
                # # main_plot_history(trials)
                # # main_plot_histogram(trials)
                # main_plot_vars(trials, do_show=False)
            
            # media = A[A.suj == suj][A.classes == (str(class_ids[0]) + ' ' + str(class_ids[1]))].acc.mean()
            # dp = A[A.suj == suj][A.classes == (str(class_ids[0]) + ' ' + str(class_ids[1]))].acc.std()
            # print(f'{round(media*100,3)}  {round(dp*100,3)} \n------------------------\n')
            
        
    pd.to_pickle(A, 'G:/Meu Drive/devs/BCI/results/trials/R4/res_as.pkl') 
    A.to_excel('G:/Meu Drive/devs/BCI/results/trials/R4/res_as.xlsx')
    R = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/R4/res_as.pkl')
    

