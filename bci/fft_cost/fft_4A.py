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
    kf = StratifiedShuffleSplit(bci.nfolds, test_size=bci.test_perc, random_state=42)  # StratifiedKFold(self.nfolds, False)
    cross_acc = []
    for train, test in kf.split(Z, t):
        ZT, tt, ZV, tv = Z[train], t[train], Z[test], t[test]
        acc_fold, _, _ = bci.sbcsp_chain(ZT, ZV, tt, tv)
        cross_acc.append(acc_fold)
    bci.acc = np.mean(cross_acc)    
    tempo.append(time()-t0)
    return bci.acc * (-1)

R = pd.DataFrame(columns=['suj','class_ids','ncsp','clog','acc_cal','acc_valid','kpa_valid','total_cost','mean_cost'])

ds = 'III4a' # 
n_iter = 50

subjects = ['aa','al','av','aw','ay']
class_ids = [2, 3]
ch = range(0,118)
Fs = 100
space = (hp.choice('ncsp', [2,4,6,8,10,18,30,42,60,78]), hp.quniform('clog', -8, 4, 1))

for suj in subjects:
    tempo = []
    data, events, info = np.load('/mnt/dados/eeg_data/' + ds + '/npy/' + str(suj) + '.npy', allow_pickle=True)
    
    bci = BCI()
    bci.class_ids = class_ids
    # bci.data, bci.events = data, events
    bci.crossval, bci.nfolds, bci.test_perc = True, 10, 0.1
    bci.tmin, bci.tmax, bci.Fs = 0.5, 2.5, Fs
    bci.is_sbcsp, bci.overlap = True, True
    bci.filtering, bci.f_low, bci.f_high, bci.nbands = {'design':'IIR', 'iir_order':5}, 4, 40, 9 # {'design':'IIR', 'iir_order':5}

    smin, smax = int(bci.tmin * Fs), int(bci.tmax * Fs)
    Z, t = extractEpochs(data, events, smin, smax, class_ids)
    kf = StratifiedShuffleSplit(1, test_size=0.5, random_state=42)
    for train, test in kf.split(Z, t): ZT, tt, ZV, tv = Z[train], t[train], Z[test], t[test]
    Z, ZT, ZV = nanCleaner(Z), nanCleaner(ZT), nanCleaner(ZV)

    trials = base.Trials()
    first = [{'ncsp': 8, 'clog': -4}]
    obj = partial(objective, bci=bci, Z=ZT, t=tt)
    
    t1 = time()
    best = fmin(obj, space=space, algo=tpe.suggest, max_evals=n_iter, trials=trials, points_to_evaluate=first)
    cost = round(time() - t1, 2)
    acc_cal = round(trials.best_trial['result']['loss']*-100,2)
    ncsp, clog = int(trials.argmin['ncsp']), int(trials.argmin['clog'])
    
    bci.ncsp, bci.clf = ncsp, {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':clog}
    bci.acc, bci.kpa, _ = bci.sbcsp_chain(ZT, ZV, tt, tv)
    acc_val, kpa_val = round(bci.acc*100,2), round(bci.kpa, 3)
    
    res = [suj, class_ids, ncsp, clog, acc_cal, acc_val, kpa_val, cost, round(np.mean(tempo),2)]
    print(res)
    R.loc[len(R)] = res
        
pd.to_pickle(R, '/home/vboas/cloud/results/cost_' + ds + '_' + bci.filtering['design'] + '_' + str(int(bci.f_low)) + '_' + str(int(bci.f_high)) + '_' + str(int(bci.nbands)) + 'sb.pkl')

print(f"Tempo iteração: {R['mean_cost'].mean()} +- {R['mean_cost'].std()}")
print(f"Tempo ciclo bo: {R['mean_cost_niter'].mean()} +- {R['mean_cost'].std()}")


########### ANALYSIS #########################################################
# DFT = pd.read_pickle('/home/vboas/cloud/results/cost_' + ds + '_FFT_' + bci.f_low + '_' + bci.f_high + '_' + bci.nbands + 'sb.pkl')
# IIR = pd.read_pickle('/home/vboas/cloud/results/cost_' + ds + '_IIR_' + bci.f_low + '_' + bci.f_high + '_' + bci.nbands + 'sb.pkl')

# print('Mean DFT:', round(DFT['acc_valid'].mean(), 2), round(DFT['kpa_valid'].mean(),3), round(DFT['mean_cost'].mean(),4))
# print('Mean IIR:', round(IIR['acc_valid'].mean(), 2), round(IIR['kpa_valid'].mean(),3), round(IIR['mean_cost'].mean(),4))

# print(f"Ganho Acc DFT: {round((DFT['acc_valid'].mean() - IIR['acc_valid'].mean())*100,2)}%")

# print('================ DFT ================')
# print(DFT.iloc[:,8:].describe())
# print('================ IIR ================')
# print(IIR.iloc[:,8:].describe())
