#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas

import mne
import pickle
import warnings
import numpy as np
import pandas as pd
from time import time
from hyperopt import base, fmin, tpe, hp
from bci_cp import Processor
from functools import partial

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

# import math
# import itertools
# from scipy.linalg import eigh
# from scipy.fftpack import fft
# from scipy.signal import lfilter, butter, iirfilter, filtfilt
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from bci_cp import extractEpochs
# data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01T.npy', allow_pickle=True)
# epochs, labels = extractEpochs(data, events, int(0.5*info['fs']), int(2.5*info['fs']), [1,2])
# Z = [epochs[np.where(labels==i)] for i in [1,2]]
# t = np.r_[1*np.ones(int(len(Z[0])/2)), 2*np.ones(int(len(Z[1])/2))]
# b, a = butter(5, [8/(0.5*info['fs']), 30/(0.5*info['fs'])], btype='bandpass')
# X = lfilter(b, a, Z[0][0]) 
# m = np.mean(X, axis=-1)
# c = X@X.T/len(X[0])

def validate(args, path_eeg, class_ids, Fs, ch):
    # print(args)
    fl, fh, tmin, tmax, ncsp, nbands, clf = args
    while(tmax-tmin) < 1: tmax += 0.5
    if int(nbands) > int(fh-fl): nbands = int(fh-fl)
    bci_as = Processor()
    bci_as.define_params(
        f_low=int(fl), f_high=int(fh), ncsp=int(ncsp), class_ids=class_ids, tmin=tmin, tmax=tmax,
        fs=Fs, filt_type='DFT', clf_dict=clf, filt_order=None, is_sbcsp=True if nbands>1 else False, 
        nbands=int(nbands) if nbands>1 else None, overlap=True, crossval=True, nfolds=5, test_perc=0.4)
    bci_as.load_eeg_train(eeg_path=path_eeg, channels=ch)
    try: bci_as.process(single_eeg=True)
    except: bci_as.acc = 0.1
    bci_as.clear_eeg_data()
    return bci_as.acc * (-1)


n_iter = 20
path = '/home/vboas/cloud/results/2020_10_17_as_off/'
class_ids = [1, 2]
datasets = ['IV2a', 'IV2b', 'Lee19']
datasets = ['IV2a'] # IV2a, IV2b, Lee19, LINCE
for ds in datasets:
    subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
    subjects = [1,2,3]
    for suj in subjects:
        sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
        path_dump = path + ds + '_trials/' + sname + str(suj) + '.pkl'
        Fs = 125 if ds == 'Lee19' else 250
        ch = range(0,22) if ds=='IV2a' else range(0,3) if ds=='IV2b' else range(0,16) if ds=='Lee19' else range(0,8)
        if ds in ['IV2a','IV2b']:
            path_eeg_train = '/mnt/dados/eeg_data/' + ds + '/npy/' + sname + '0' + str(suj) + 'T.npy'
            path_eeg_test = '/mnt/dados/eeg_data/' + ds + '/npy/' + sname + '0' + str(suj) + 'E.npy'
        elif ds == 'Lee19':
            path_eeg_train = '/mnt/dados/eeg_data/' + ds + '/npy16/S' + str(suj) + 'T.npy'
            eeg_path_test = '/mnt/dados/eeg_data/' + ds + '/npy16/S' + str(suj) + 'E.npy'
        else:
            path_eeg_train = '/mnt/dados/eeg_data/' + ds + '/npy/' + str(suj) + '_S1.npy'
            path_eeg_test = '/mnt/dados/eeg_data/' + ds + '/npy/' + str(suj) + '_S2.npy'   
        print(f'###### {sname + str(suj)} [1,2] ######')
    
        # ============
        # SB approach
        # ============
        bci_sb = Processor()
        bci_sb.define_params(f_low=4, f_high=40, ncsp=8, class_ids=class_ids, tmin=0.5, tmax=2.5, fs=Fs, filt_type='IIR', filt_order=5,
                             clf_dict={'model':'SVM','kernel':{'kf':'linear'},'C':-4}, is_sbcsp=True, nbands=9, overlap=True, crossval=False)
        bci_sb.load_eeg_train(eeg_path=path_eeg_train, channels=ch)
        bci_sb.load_eeg_test(eeg_path=path_eeg_train, channels=ch)
        bci_sb.process()
        bci_sb.acc_cal = bci_sb.acc
        
        bci_sb.load_eeg_train(eeg_path=path_eeg_train, channels=ch)
        bci_sb.load_eeg_test(eeg_path=path_eeg_test, channels=ch)
        bci_sb.process()
        bci_sb.clear_eeg_data()
    
        # ============
        # BU approach
        # ============
        bci_bu = Processor()
        bci_bu.define_params(f_low=8, f_high=30, ncsp=8, class_ids=class_ids, tmin=0.5, tmax=2.5, fs=Fs, filt_type='IIR', filt_order=5,
                             clf_dict={'model':'LDA'}, is_sbcsp=False, nbands=None, crossval=False)
        bci_bu.load_eeg_train(eeg_path=path_eeg_train, channels=ch)
        bci_bu.load_eeg_test(eeg_path=path_eeg_train, channels=ch)
        bci_bu.process()
        bci_bu.acc_cal = bci_bu.acc
        
        bci_bu.load_eeg_train(eeg_path=path_eeg_train, channels=ch)
        bci_bu.load_eeg_test(eeg_path=path_eeg_test, channels=ch)
        bci_bu.process()
        bci_bu.clear_eeg_data()
    
        # ============
        # AS approach
        # ============
        space = (
            hp.uniformint('fl', 0, 15),
            hp.uniformint('fh', 25, 50),
            hp.quniform('tmin', 0, 2, 0.5),
            hp.quniform('tmax', 2, 4, 0.5),
            hp.choice('ncsp', [2, 4, 6, 8, len(ch)]),
            hp.uniformint('nbands', 1, 25),
            hp.choice('clf', [
                {'model':'LR'},
                {'model':'LDA'},
                {'model':'KNN', 'neig':hp.uniformint('neig', 2, 20)}, #'metric':hp.choice('metric', ['euclidean', 'manhattan', 'minkowski', 'chebyshev'])},
                {'model':'SVM', 'C':hp.quniform('C', -6, 0, 1), 'kernel':hp.choice('kernel', [{'kf':'linear'}, {'kf':'poly'}, {'kf':'sigmoid'}, {'kf':'rbf'}])},
                {'model':'MLP', 'eta':hp.quniform('eta', -5, -2, 1), 'n_neurons':hp.quniform('n_neurons', 20, 100, 20),  
                 'n_hidden':hp.uniformint('n_hidden', 1, 2)}, # 'activ':hp.choice('activ', [{'af':'logistic'}, {'af':'tanh'}])}
                ]))
    
        trials = base.Trials()
        try: trials = pickle.load(open(path_dump, 'rb'))
        except: trials = base.Trials()
        try:
            print('Past trials: ' + str(len(trials)))
            init_vals = [{'fl':4, 'fh':40, 'tmin':0.5, 'tmax':2.5, 'ncsp':8, 'nbands':9, 'clf':{'model':'SVM', 'C':-4, 'kf':'linear'}}]
            validate2 = partial(validate, path_eeg=path_eeg_train, class_ids=class_ids, Fs=Fs, ch=ch)
            best = fmin(validate2, space=space, algo=tpe.suggest, max_evals=len(trials)+n_iter, trials=trials, points_to_evaluate=init_vals)
            pickle.dump(trials, open(path_dump, 'wb'))
        except:
            pickle.dump(trials, open(path_dump, 'wb'))
            raise
    
        # best = trials.best_trial['misc']['vals']
        # best = trials.argmin
        if best['clf'] == 0: clf = {'model':'LR'}
        elif best['clf'] == 1: clf = {'model':'LDA'}
        elif best['clf'] == 2: clf = {'model':'KNN', 'neig':best['neig']}
            # metric = 'euclidean' if best['metric']==0 else 'manhattan' if best['metric']==1 else 'minkowski' if best['metric']==2 else 'chebyshev' 
            # clf = {'model':'KNN', 'metric':metric, 'neig':best['neig']}
        elif best['clf'] == 3:
            kernel = 'linear' if best['kernel']==0 else 'poly' if best['kernel']==1 else 'sigmoid' if best['kernel']==2 else 'rbf'
            clf = {'model':'SVM', 'kernel':{'kf':kernel}, 'C':best['C']}
        elif best['clf'] == 4: clf = {'model':'MLP', 'eta':best['eta'], 'n_neurons':best['n_neurons'], 'n_hidden':best['n_hidden']}
            # activ = 'logistic' if best['activ']==0 else 'tanh'
            # clf = {'model':'MLP', 'eta':best['eta'], 'activ':{'af':activ}, 'n_neurons':best['n_neurons'], 'n_hidden':best['n_hidden']}
        
        while(best['tmax']-best['tmin']) < 1: best['tmax'] += 0.5
        if best['nbands'] > (best['fh']-best['fl']): best['nbands'] = best['fh']-best['fl'] 
         
        bci_as = Processor()
        bci_as.define_params(f_low=int(best['fl']), f_high=int(best['fh']), ncsp=[2,4,6,8,len(ch)][best['ncsp']], class_ids=class_ids, 
                             tmin=best['tmin'], tmax=best['tmax'], fs=Fs, filt_type='DFT', filt_order=None,
                             clf_dict=clf, is_sbcsp=True if best['nbands']>1 else False, 
                             nbands=int(best['nbands']) if best['nbands']>1 else None, crossval=False)
        bci_as.load_eeg_train(eeg_path=path_eeg_train, channels=ch)
        bci_as.load_eeg_test(eeg_path=path_eeg_train, channels=ch)
        bci_as.process()
        bci_as.acc_cal = bci_as.acc
        
        bci_as.load_eeg_train(eeg_path=path_eeg_train, channels=ch)
        bci_as.load_eeg_test(eeg_path=path_eeg_test, channels=ch)
        bci_as.process()
        bci_as.clear_eeg_data()
        
        # ============
        # COST
        # ============
        # cost_iir = []
        # cost_dft = []
        # trials = pickle.load(open(path_dump, 'rb'))
        # for i in range(len(trials)):
        #     h = trials.miscs[i]['vals']
        #     if h['clf'][0] == 0: clf = {'model':'LR'}
        #     elif h['clf'][0] == 1: clf = {'model':'LDA'}
        #     elif h['clf'][0] == 2: clf = {'model':'KNN', 'neig':h['neig'][0]}
        #     elif h['clf'][0] == 3:
        #         kernel = 'linear' if h['kernel'][0]==0 else 'poly' if h['kernel'][0]==1 else 'sigmoid' if h['kernel'][0]==2 else 'rbf'
        #         clf = {'model':'SVM', 'kernel':{'kf':kernel}, 'C':h['C'][0]}
        #     elif h['clf'][0] == 4: clf = {'model':'MLP', 'eta':h['eta'][0], 'n_neurons':h['n_neurons'][0], 'n_hidden':h['n_hidden'][0]}
        #     while(h['tmax'][0]-h['tmin'][0]) < 1: h['tmax'][0] += 0.5
        #     if h['nbands'][0] > (h['fh'][0]-h['fl'][0]): h['nbands'][0] = h['fh'][0]-h['fl'][0] 
    
        #     bci = Processor()
        #     bci.define_params(f_low=h['fl'][0], f_high=h['fh'][0], ncsp=h['ncsp'][0], class_ids=class_ids, 
        #                       tmin=h['tmin'][0], tmax=h['tmax'][0], fs=Fs, filt_type='DFT', filt_order=None,
        #                       clf_dict=clf, is_sbcsp=True if h['nbands'][0]>1 else False, 
        #                       nbands=int(h['nbands'][0]) if h['nbands'][0]>1 else None, 
        #                       overlap=True, crossval=True, nfolds=5, test_perc=0.4)
        #     bci.load_eeg_train(eeg_path=path_eeg_train, channels=ch)
        #     t0 = time()
        #     try: bci.process(single_eeg=True)
        #     except: bci.acc = 0.1
        #     cost_dft.append(time()-t0)
            
        #     bci.filt_type, bci.filt_order = 'IIR', 5
        #     t0 = time()
        #     try: bci.process(single_eeg=True)
        #     except: bci.acc = 0.1
        #     cost_iir.append(time()-t0)
        #     bci.clear_eeg_data()
        
        # ============
        # SHOW RESULTS
        # ============
        print('')
        print(f"Setup: {bci_as.tmin}-{bci_as.tmax}s {bci_as.f_low}-{bci_as.f_high}Hz Ns={bci_as.nbands} Ncsp={bci_as.ncsp} CLF={bci_as.clf_dict}")
        print(f'C bu={round(bci_bu.acc_cal*100,2)} sb={round(bci_sb.acc_cal*100,2)} as={round(bci_as.acc_cal*100,2)}')
        print(f'V bu={round(bci_bu.acc*100,2)} sb={round(bci_sb.acc*100,2)} as={round(bci_as.acc*100,2)}')    
        # print(f"cost_dft={round(np.mean(cost_dft),2)}+{round(np.std(cost_dft),2)} cost_iir={round(np.mean(cost_iir),2)}+{round(np.std(cost_iir),2)}")
    
        #### Carregando resultados de testes antigos
        # l = Processor()
        # l.load_setup('/home/vboas/cloud/results/as_off/' + ds + '/' + sname + str(suj) + '_learner')
        # print('')
        # print(f"Setup: {l.tmin}-{l.tmax}s {l.f_low}-{l.f_high}Hz Ns={l.nbands} {l.ncsp_list} Ncsp={l.ncsp} CLF={l.clf_dict}")
        # print(f'C bu={round(l.learner_cla.acc_cal*100,2)} sb={round(l.learner_sb.acc_cal*100,2)} as={round(l.acc_cal*100,2)}')
        # print(f'V bu={round(l.learner_cla.acc*100,2)} sb={round(l.learner_sb.acc*100,2)} as={round(l.acc*100,2)}')
    
        # ============
        # SAVE RESULTS
        # ============
        R = pd.Series(data=[sname+str(suj), len(ch), bci_as.f_low, bci_as.f_high, bci_as.tmin, bci_as.tmax, bci_as.nbands, bci_as.ncsp, bci_as.clf_dict['model'], bci_as.clf_dict,
                         bci_as.acc, bci_bu.acc, bci_sb.acc, bci_as.acc_cal, bci_bu.acc_cal, bci_sb.acc_cal
                         ]) #, round(np.mean(cost_dft),2), round(np.mean(cost_iir),2)])
        pd.to_pickle(R, path + 'R_' + sname + str(suj) + '.pkl')

        
# # =================================
# # SAVE RESULTS (all subjects in ds)
# # =================================
# datasets = ['IV2a', 'IV2b', 'Lee19']
# datasets = ['IV2a'] # IV2a, IV2b, Lee19, LINCE
# for ds in datasets:
#     subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
#     for suj in subjects:
#         sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
#     RDS = pd.DataFrame(columns=['subj', 'nchannels', 'fl', 'fh', 'tmin', 'tmax', 'nbands', 'ncsp', 'clf', 'clf_details', 
#                                 'as_acc', 'cla_acc', 'sb_acc', 'as_acc_cal', 'cla_acc_cal', 'sb_acc_cal', 
#                                 ]) #, 'fft_as_mcost','iir_as_mcost'])
#     for suj in subjects: RDS.loc[len(RDS)] = pd.Series.tolist(pd.read_pickle(path + 'R_' + sname + str(suj) + '.pkl'))
#     pd.to_pickle(RDS, path + 'R_' + ds + '.pkl')

# # ===========================
# # SAVE RESULTS (all datasets)
# # ===========================
# pd.to_pickle(pd.concat([pd.read_pickle(path + 'R_' + ds + '.pkl') for ds in ['IV2a', 'IV2b', 'Lee19'] ], ignore_index=True), path + 'RFull.pkl')


# ============
# LOAD RESULTS
# ============
# ds, suj = 'IV2a', 1
# sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''

# R = pd.read_pickle(path + 'R_' + sname + str(suj) + '.pkl')
# print(R[['as_acc','cla_acc','sb_acc']].mean()*100)

# RDS = pd.read_pickle(path + 'R_' + ds + '.pkl')
# print(RDS[['as_acc','cla_acc','sb_acc']].mean()*100)

# RFull = pd.read_pickle(path + 'RFull.pkl')
# print(RFull[['as_acc','cla_acc','sb_acc']].mean()*100)


# subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
# RDS = pd.read_pickle(path + 'R_' + ds + '.pkl')
# for suj in subjects:
#     sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
#     R = RDS[RDS['subj'] == (sname + str(suj))].iloc[0]
#     print('')
#     print(f'>>> {sname + str(suj)} <<< ')
#     print(f"Setup: {R['tmin']}-{R['tmax']}s {R['fl']}-{R['fh']}Hz Ns={R['nbands']} Ncsp={R['ncsp']} CLF={R['clf_details']}")
#     print(f"C bu={round(R['bu_acc_cal']*100,2)} sb={round(R['sb_acc_cal']*100,2)} as={round(R['as_acc_cal']*100,2)}")
#     print(f"V bu={round(R['bu_acc']*100,2)} sb={round(R['sb_acc']*100,2)} as={round(R['as_acc']*100,2)}")    
#     # print(f"cost_dft={round(np.mean(cost_dft),2)}+{round(np.std(cost_dft),2)} cost_iir={round(np.mean(cost_iir),2)}+{round(np.std(cost_iir),2)}")

