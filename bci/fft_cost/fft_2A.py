# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from hyperopt import base, fmin, tpe, hp
from time import time
from functools import partial
from scripts.bci_cp import Processor

t = []
def objective(args, bci):
    # print(args)
    ncsp, clog = args
    bci.clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':clog}
    bci.ncsp = int(ncsp)
    # bci.mbands = nbands
    t0 = time()
    try: bci.process(single_eeg=True)
    except: bci.acc = 0.1
    t.append(time()-t0)
    return bci.acc * (-1)

R = pd.DataFrame(columns=['suj','class_ids','ncsp','clog','acc_cal','acc_valid','kpa_valid','total_cost','mean_cost'])

ds = 'IV2a' # III4a
n_iter = 50
        
subjects = range(1,10) 
classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
ch = range(0,22)
Fs = 250
space = (hp.quniform('ncsp', 2, len(ch), 2), hp.quniform('clog', -8, 4, 1))
    
# space = (
#     # hp.uniformint('fl', 0, 15),
#     # hp.uniformint('fh', 20, 45),
#     # hp.uniformint('nbands', 1, 25),
#     hp.quniform('ncsp', 2, len(ch), 2),
#     # hp.choice('ncsp', [2,4,6,8,22]),
#     hp.quniform('clog', -8, 4, 1),
# )

for suj in subjects:
    for class_ids in classes:
        t = []
        bci = Processor()
        bci.class_ids = class_ids
        bci.load_eeg_train(eeg_path='/mnt/dados/eeg_data/' + ds + '/npy/A0' + str(suj) + 'T.npy', channels=ch)
        bci.load_eeg_test(eeg_path='/mnt/dados/eeg_data/' + ds + '/npy/A0' + str(suj) + 'E.npy', channels=ch)
        bci.crossval, bci.nfolds, bci.test_perc = True, 10, 0.1
        bci.tmin, bci.tmax, bci.Fs = 0.5, 2.5, Fs
        bci.is_sbcsp, bci.overlap = True, True 
        bci.filt_type, bci.filt_order, bci.f_low, bci.f_high, bci.nbands = 'IIR', 5, 4, 40, 9
        
        trials = base.Trials()
        first = [{'ncsp': 8, 'clog': -4}]
        obj = partial(objective, bci=bci)
        
        t1 = time()
        best = fmin(obj, space=space, algo=tpe.suggest, max_evals=n_iter, trials=trials, points_to_evaluate=first)
        cost = round(time() - t1, 2)
        acc_cal = round(trials.best_trial['result']['loss']*-100,2)
        ncsp, clog = int(trials.argmin['ncsp']), int(trials.argmin['clog'])
        
        bci.ncsp, bci.clf = ncsp, {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':clog}
        bci.crossval = False
        bci.process(single_eeg=False)
        acc_val, kpa_val = round(bci.acc*100,2), round(bci.kpa, 3)
        
        res = [suj, class_ids, ncsp, clog, acc_cal, acc_val, kpa_val, cost, round(np.mean(t),2)]
        print(res)
        R.loc[len(R)] = res
        
pd.to_pickle(R, '/home/vboas/cloud/results/cost_' + ds + '_' + bci.filt_type + '_' +  str(int(bci.f_low)) + '_' + str(int(bci.f_high)) + '_' + str(int(bci.nbands)) + 'sb.pkl')


########### ANALYSIS #########################################################
DFT = pd.read_pickle('/home/vboas/cloud/results/cost_' + ds + '_FFT_' + bci.f_low + '_' + bci.f_high + '_' + bci.nbands + 'sb.pkl')
IIR = pd.read_pickle('/home/vboas/cloud/results/cost_' + ds + '_IIR_' + bci.f_low + '_' + bci.f_high + '_' + bci.nbands + 'sb.pkl')

print('Mean DFT:', round(DFT['acc_valid'].mean(), 2), round(DFT['kpa_valid'].mean(),3), round(DFT['mean_cost'].mean(),4))
print('Mean IIR:', round(IIR['acc_valid'].mean(), 2), round(IIR['kpa_valid'].mean(),3), round(IIR['mean_cost'].mean(),4))

print(f"Ganho Acc DFT: {round((DFT['acc_valid'].mean() - IIR['acc_valid'].mean())*100,2)}%")

print('================ DFT ================')
print(DFT.iloc[:,8:].describe())
print('================ IIR ================')
print(IIR.iloc[:,8:].describe())
