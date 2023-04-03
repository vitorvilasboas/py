# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from hyperopt import base, fmin, tpe, hp
from time import time
from functools import partial
from scripts.bci_cp import Processor
from scripts.bci_utils import BCI, extractEpochs, nanCleaner
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, StratifiedKFold

tempo = []
def objective(args, bci, Z, t): # class_ids, suj):
    # print(args)
    ncsp, clog = args
    bci.clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':clog}
    bci.ncsp = int(ncsp)
    
    t0 = time()
    # kf = StratifiedShuffleSplit(10, test_size=0.1, random_state=42)  # StratifiedKFold(10, False)
    kf = StratifiedShuffleSplit(1, test_size=0.2, random_state=42)
    cross_acc = []
    for train, test in kf.split(Z, t):
        ZT1, tt1, ZV1, tv1 = Z[train], t[train], Z[test], t[test]
        acc_fold, _, _ = bci.sbcsp_chain(ZT1, ZV1, tt1, tv1)
        cross_acc.append(acc_fold)
    acc = np.mean(cross_acc)
    tempo.append(time()-t0)
    return acc * (-1)

R = pd.DataFrame(columns=['suj','class_ids','ncsp','clog','acc_cal','acc_valid','kpa_valid','total_cost','mean_cost',
                          'dp_cost','dp_cal','dp_val','dp_kpa','total_treino','mean_cost_niter','dp_cost_niter'])

ds = 'IV2a' # 
n_iter = 50

subjects = range(1,10) 
classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
ch = range(0,22)
Fs = 250

space = (hp.quniform('ncsp', 2, len(ch), 2), hp.quniform('clog', -8, 4, 1))
    
for suj in subjects:
    for class_ids in classes:
        tempo = []
        data, events, info = np.load('/mnt/dados/eeg_data/' + ds + '/npy/A0' + str(suj) + '.npy', allow_pickle=True)
        data = data[range(0,22)]
        tmin, tmax = 0.5, 2.5
        smin, smax = int(tmin * Fs), int(tmax * Fs)
        Z, t = extractEpochs(data, events, smin, smax, class_ids)
        Z = nanCleaner(Z)
        # print(np.asarray(Z).shape)
        
        bci = BCI()
        # bci.data, bci.events = data, events
        bci.class_ids = class_ids
        bci.tmin, bci.tmax, bci.Fs = tmin, tmax, Fs
        bci.is_sbcsp, bci.overlap = True, True
        bci.crossval, bci.nfolds, bci.test_perc = True, 10, 0.1
        bci.filtering, bci.f_low, bci.f_high, bci.nbands = {'design':'DFT'}, 0, 50, 24 # {'design':'IIR', 'iir_order':5}
    
        T = []
        
        kf = StratifiedShuffleSplit(bci.nfolds, test_size=bci.test_perc, random_state=42)
        cv_cal, cv_acc, cv_kpa = [], [], []
        for train, test in kf.split(Z, t):
            t1 = time()
            ZT, tt, ZV, tv = Z[train], t[train], Z[test], t[test]
            # print(np.asarray(ZT).shape)
            
            trials = base.Trials()
            first = [{'ncsp': 8, 'clog': -4}]
            obj = partial(objective, bci=bci, Z=ZT, t=tt)
        
            best = fmin(obj, space=space, algo=tpe.suggest, max_evals=n_iter, trials=trials, points_to_evaluate=first)
            
            cv_cal.append(trials.best_trial['result']['loss']*(-1))
            ncsp, clog = int(trials.argmin['ncsp']), int(trials.argmin['clog'])
        
            bci.ncsp, bci.clf = ncsp, {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':clog}
            a, k, _ = bci.sbcsp_chain(ZT, ZV, tt, tv)
            cv_acc.append(a) 
            cv_kpa.append(k) 
            T.append(time() - t1)
        
        acc_cal, acc_val, kpa_val = round(np.mean(cv_cal)*100,2), round(np.mean(cv_acc)*100,2), round(np.mean(cv_kpa), 3)
        dp_cal, dp_val, dp_kpa = round(np.std(cv_cal)*100,2), round(np.std(cv_acc)*100,2), round(np.std(cv_kpa), 3)
        
        mean_cost, dp_cost = round(np.mean(T),2), round(np.std(T),2)
        
        mean_niter, dp_niter = round(np.mean(tempo),2), round(np.std(tempo),2)
    
        res = [suj, class_ids, ncsp, clog, acc_cal, acc_val, kpa_val, sum(T), mean_cost, 
           dp_cost, dp_cal, dp_val, dp_kpa, sum(tempo), mean_niter, dp_niter] #  
        # print(cost/(n_iter*bci.nfolds*1))
        
        # A = {'acc_cal':cv_cal, 'acc_val':cv_acc, 'kpa_val':cv_kpa}
            
pd.to_pickle(R, '/home/vboas/cloud/results/cost_' + ds + '_' + bci.filtering['design'] + '_' + str(int(bci.f_low)) + '_' + str(int(bci.f_high)) + '_' + str(int(bci.nbands)) + 'sb.pkl')

########### ANALYSIS #########################################################
# DFT = pd.read_pickle('/home/vboas/cloud/results/cost_' + ds + '_DFT_' + str(int(bci.f_low)) + '_' + str(int(bci.f_high)) + '_' + str(int(bci.nbands)) + 'sb.pkl')
# IIR = pd.read_pickle('/home/vboas/cloud/results/cost_' + ds + '_IIR_' + str(int(bci.f_low)) + '_' + str(int(bci.f_high)) + '_' + str(int(bci.nbands)) + 'sb.pkl')

# print('ACC DFT:', round(DFT['acc_valid'].mean(), 2), round(DFT['dp_val'].mean(),3))
# print('ACC IIR:', round(IIR['acc_valid'].mean(), 2), round(IIR['dp_val'].mean(),3))

# print('COSTS DFT:', round(DFT['mean_cost'].mean(),2), round(DFT['dp_cost'].mean(),1), round(DFT['mean_cost_niter'].mean(),2), round(DFT['dp_cost_niter'].mean(),2))
# print('COSTS IIR:', round(IIR['mean_cost'].mean(),2), round(IIR['dp_cost'].mean(),1), round(IIR['mean_cost_niter'].mean(),2), round(IIR['dp_cost_niter'].mean(),2))


# # print(f"Ganho Acc DFT: {round((DFT['acc_valid'].mean() - IIR['acc_valid'].mean())*100,2)}%")

# # print('================ DFT ================')
# # print(DFT.iloc[:,8:].describe())
# # print('================ IIR ================')
# # print(IIR.iloc[:,8:].describe())
