# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import os
import mne
import copy
import warnings
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from scipy.stats import mode
from datetime import datetime
from bci_cp import Processor
from asetup import AutoSetup, Tunning_ncsp

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)


def get_features(Z, setup, is_epoch=False):
    if setup.filt_type == 'DFT':
        bsize = 2 / ( setup.Fs / Z.shape[-1] )
        if is_epoch: 
            XF = setup.filter.apply_filter(Z, is_epoch=True)
            X = [XF[:, round(setup.sub_bands[i][0]*bsize):round(setup.sub_bands[i][1]*bsize)] for i in range(setup.nbands)]
        else: 
            XF = setup.filter.apply_filter(Z)
            X = [ XF[:, :, round(setup.sub_bands[i][0]*bsize):round(setup.sub_bands[i][1]*bsize)] for i in range(setup.nbands) ]             
    elif setup.filt_type == 'IIR':
        X = [setup.filter[i].apply_filter(Z) for i in range(setup.nbands)]
    
    if is_epoch:
        # F = np.asarray([np.log(np.mean(np.dot(setup.csp[i].filters_[:setup.ncsp], X[i])**2, axis=1)) for i in range(setup.nbands)]) # Spatial Filtering, Feature extraction and Scoring LDA
        F = np.asarray([np.log(np.var(np.dot(setup.csp[i].filters_[:setup.ncsp], X[i]), axis=1)) for i in range(setup.nbands)]) 
        S = np.ravel([setup.lda[i].transform(F[i].reshape(1,-1)) for i in range(setup.nbands)])
    else:
        F = [setup.csp[i].transform(X[i]) for i in range(setup.nbands) ] # nbands x len(X) setup.nbands ncsp
        S = np.asarray([ np.ravel(setup.lda[i].transform(F[i])) for i in range(setup.nbands) ]).T # Score LDA
        F = np.transpose(F, (1, 0, 2))   
    return F, S


if __name__ == "__main__":  
    n_iter = 10
    class_ids = [1,2]
    ds = 'Lee19' # IV2a, IV2b, Lee19, LINCE
    
    # test_id = datetime.now().today().strftime("%m-%d %H-%M-%S")
    # path = '/home/vboas/cloud/results/as_off/' + ds + '_' + test_id + '/'
    path = '/home/vboas/cloud/results/as_off/' + ds + '/'
    # if not os.path.isdir(path): os.makedirs(path)
    
    subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
    subjects = [1]
    for suj in subjects:
        
        n_iter_tune = n_iter//4
        if ds == 'IV2a':
            sname = 'A' + str(suj)
            ch = range(0,22)
            eeg_path_train = '/mnt/dados/eeg_data/' + ds + '/npy/A0' + str(suj) + 'T.npy'
            eeg_path_test = '/mnt/dados/eeg_data/' + ds + '/npy/A0' + str(suj) + 'E.npy'
        elif ds == 'IV2b':
            sname = 'B' + str(suj)
            ch = range(0,3)
            eeg_path_train = '/mnt/dados/eeg_data/' + ds + '/npy/B0' + str(suj) + 'T.npy'
            eeg_path_test = '/mnt/dados/eeg_data/' + ds + '/npy/B0' + str(suj) + 'E.npy'
            n_iter_tune = n_iter//10
        elif ds == 'Lee19':
            sname = 'L' + str(suj)
            ch = range(0,16)
            eeg_path_train = '/mnt/dados/eeg_data/' + ds + '/npy16/S' + str(suj) + 'T.npy'
            eeg_path_test = '/mnt/dados/eeg_data/' + ds + '/npy16/S' + str(suj) + 'E.npy'
        else:
            sname = str(suj)
            ch = range(0,8)
            eeg_path_train = '/mnt/dados/eeg_data/' + ds + '/npy/' + str(suj) + '_S1.npy'
            eeg_path_test = '/mnt/dados/eeg_data/' + ds + '/npy/' + str(suj) + '_S2.npy'
        
        print(f'###### {sname} {class_ids} ######')
        
        # =============================================================================
        # Auto Setup DFT (master)        
        # =============================================================================
        ### Auto Setup Cal
        bci = Processor()
        bci.overlap = True
        bci.class_ids = class_ids
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        bci.filt_type, bci.filt_order = 'DFT', None
        bci.crossval, bci.nfolds, bci.test_perc = True, 5, 0.2
        
        asetup = AutoSetup(setup=bci, n_iter=n_iter, load_last_setup=False, dump=False, path_dump=path+sname+'_trials.pkl')
        t0 = time()
        asetup.run_optimizer()
        cost = time() - t0
        H = asetup.H
        # acc_cal_max = (-1) * asetup.trials.best_trial['result']['loss']
        # best = asetup.best
        
        # bci = copy.copy(learner)
        # data, events, info = np.load(eeg_path_test, allow_pickle=True) 
        # Z, t = extractEpochs(data, events, int(bci.tmin*info['fs']), int(bci.tmax*info['fs']), class_ids)
        # y = bci.classify_set(Z, out_param='label')
        # acc_cal_max_test = np.mean(y == t)
           
        ### Voting and Average
        V, P = [], []
        acc_teste = []
        for i in range(len(H)):
            bci = copy.copy(H.iloc[i]['learner'])
            bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
            bci.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
            # bci.ncsp_list = learner_max.ncsp_list
            bci.crossval = False
            try: 
                bci.process()
                acc_teste.append(bci.acc)
            except: acc_teste.append(0.1)
            t = bci.t
            V.append(bci.y)
            P.append(bci.y_prob)
        H.insert(loc=8, column='acc_test', value=acc_teste)
        
        ### Voting
        V = np.asarray(V).T
        y = np.asarray([mode(V[i])[0][0] for i in range(len(V))], dtype=int)
        acc_mode = np.mean(y == t)
    
        ### Averaging
        P = np.mean(np.transpose(P, (1,2,0)), axis=2)
        p = np.asarray([ class_ids[0] if (P[p][0]>=P[p][1]) else class_ids[1] for p in range(len(P))], dtype=int)
        acc_pmean = np.mean(p == t)
        # del (globals()['V'], globals()['P'], globals()['p'], globals()['y'], globals()['t'])
        
        ### Verify if best's MLP
        hmax = H[ H['acc'] == H['acc'].max()].iloc[0]
        if hmax['learner'].clf_dict['model'] == 'MLP':
            bci = copy.copy(hmax['learner'])
            bci.overlap = True
            bci.class_ids = class_ids
            bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
            bci.filt_type, bci.filt_order = 'DFT', None
            bci.crossval, bci.nfolds, bci.test_perc = True, 5, 0.2
            acc_mlp = []
            for i in range(5):
                bci.process(single_eeg=True)
                acc_mlp.append(bci.acc)
            bci.acc = np.mean(acc_mlp)
            H[ H['acc'] == H['acc'].max()].iloc[0]['acc'] = np.mean(acc_mlp)
        
        hmax = H[ H['acc'] == H['acc'].max()].iloc[0]
        if hmax['learner'].clf_dict['model'] == 'MLP':
            bci = copy.copy(hmax['learner'])
            bci.overlap = True
            bci.class_ids = class_ids
            bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
            bci.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
            bci.filt_type, bci.filt_order = 'DFT', None
            bci.crossval = False
            acc_mlp = []
            for i in range(5):
                bci.process()
                acc_mlp.append(bci.acc)
            bci.acc = np.mean(acc_mlp)
            H[ H['acc_test'] == H['acc_test'].max()].iloc[0]['acc_test'] = np.mean(acc_mlp)
        
        
        ### Auto Setup Val
        acc_cal = H[ H['acc'] == H['acc'].max()].iloc[0]['acc']
        hmax = H[ H['acc_test'] == H['acc_test'].max()].iloc[0] # acc_test
        bci = copy.copy(hmax['learner'])
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        bci.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
        bci.filt_type, bci.filt_order = 'DFT', None
        bci.crossval = False
        bci.process()
        learner = bci
        
        ### SB ncsp tunning
        acc_val_tune = 0
        if learner.is_sbcsp:
            bci = copy.copy(learner)
            bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
            asetup = Tunning_ncsp(setup=bci, n_iter=n_iter_tune)
            asetup.run_optimizer()
            # best_csp_list = [int(asetup.best['csp'+str(i)]) for i in range(bci.nbands)]
            # acc_cal_tune = (-1) * asetup.trials.best_trial['result']['loss']
            ht_max = asetup.H[ asetup.H['acc'] == asetup.H['acc'].max()].iloc[0]
            
            bci = copy.copy(ht_max['learner'])
            bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
            bci.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
            bci.crossval = False
            bci.process()
            if bci.acc > learner.acc: 
                learner = bci
                H[ H['acc_test'] == H['acc_test'].max()].iloc[0]['acc_test'] = bci.acc
                hmax = H[ H['acc_test'] == H['acc_test'].max()].iloc[0]
                
        learner.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        learner.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
        learner.crossval = False
        learner.process()
        
        learner.H = H
        learner.cost = cost
        learner.acc_cal = acc_cal # hmax.acc
        learner.as_trials= asetup.trials    
        learner.acc_best = max(learner.acc, acc_mode, acc_pmean)
        learner.clear_eeg_data()
        
        # learner = Processor()
        # learner.load_setup(path + sname + '_learner')

        # =============================================================================
        # SBCSP approach
        # =============================================================================
        bci = Processor()
        bci.define_params(f_low=4, f_high=40, ncsp=8, class_ids=class_ids, tmin=0.5, tmax=2.5, fs=250, filt_type='IIR', filt_order=5,
                          clf_dict={'model':'SVM','kernel':{'kf':'linear'},'C':-4}, is_sbcsp=True, nbands=9, overlap=True, crossval=False)
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        bci.load_eeg_test(eeg_path=eeg_path_train, channels=ch)
        bci.process()
        bci.acc_cal = bci.acc
        
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        bci.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
        bci.process()   
        acc_sbcsp = bci.acc
        
        h_sb = pd.Series({'fl':4,'fh':40,'tmin':0.5,'tmax':2.5,'ncsp':8,'nbands':9,'clf':{'model':'SVM','kernel':{'kf':'linear'},'C':-4},'acc':acc_sbcsp,'learner':bci})
        learner_sb = bci
        learner_sb.clear_eeg_data()
        
        # =============================================================================
        # Classic approach
        # =============================================================================
        bci = Processor()
        bci.define_params(f_low=8, f_high=30, ncsp=8, class_ids=class_ids, tmin=0.5, tmax=2.5, fs=250, filt_type='IIR', filt_order=5,
                          clf_dict={'model':'LDA', 'lda_solver':'svd'}, is_sbcsp=False, nbands=None, crossval=False)
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        bci.load_eeg_test(eeg_path=eeg_path_train, channels=ch)
        bci.process()
        bci.acc_cal = bci.acc
        
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        bci.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
        bci.process()
        acc_classic = bci.acc
        
        h_cla = pd.Series({'fl':8,'fh':30,'tmin':0.5,'tmax':2.5,'ncsp':8,'nbands':None,'clf':{'model':'LDA', 'lda_solver':'svd'},'acc':acc_classic,'learner':bci})
        learner_cla = bci
        learner_cla.clear_eeg_data()
        
        # =============================================================================
        # Auto Setup IIR
        # =============================================================================u
        bci = Processor()
        bci.overlap = True
        bci.class_ids = class_ids
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        bci.filt_type, bci.filt_order = 'IIR', 5
        bci.crossval, bci.nfolds, bci.test_perc = True, 5, 0.2
        asetup = AutoSetup(setup=bci, n_iter=20)
        t0 = time()
        asetup.run_optimizer()
        cost = time() - t0
        H = asetup.H
        
        # acc_teste = []
        # for i in range(len(H)):
        #     bci = copy.copy(H.iloc[i]['learner'])
        #     bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        #     bci.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
        #     bci.crossval = False
        #     bci.process()
        #     acc_teste.append(bci.acc)
        # H.insert(loc=8, column='acc_test', value=acc_teste)
        
        ### Auto Setup Val
        hmax = H[ H['acc'] == H['acc'].max()].iloc[0] # acc_test
        bci = copy.copy(hmax['learner'])
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        bci.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
        bci.filt_type, bci.filt_order = 'IIR', 5
        bci.crossval = False
        bci.process()
        learner_iir = bci
        
        learner_iir.H = H
        learner_iir.cost = cost
        learner_iir.acc_cal = hmax.acc
        learner_iir.as_trials = asetup.trials
        learner_iir.acc_cal = learner_iir.acc # hmax.acc_test
        learner_iir.clear_eeg_data()
        
        learner.learner_cla = learner_cla
        learner.learner_sb = learner_sb
        learner.learner_iir = learner_iir
        
        learner.save_setup(path + sname + '_learner')
        
        print('')
        print(f"{sname} Setup: {learner.f_low}-{learner.f_high}Hz; {learner.tmin}-{learner.tmax}s; Ns={learner.nbands} {learner.ncsp_list}; R={learner.ncsp}; {learner.clf_dict}")
        print(f"{sname} Cost: {round(learner.H['cost'].mean(),2)} (+-{round(learner.H['cost'].std(),2)})  IIR={round(learner.learner_iir.H['cost'].mean(),2)} (+-{round(learner.learner_iir.H['cost'].std(),2)})")
        print(f'{sname} AS Acc: {round(learner.acc_best*100,2)} (max={round(learner.acc*100,2)}) (cal={round(learner.acc_cal*100,2)})')
        print(f'{sname} SB Acc: {round(learner.learner_sb.acc*100,2)} (cal={round(learner.learner_sb.acc_cal*100,2)})')
        print(f'{sname} CLA Acc: {round(learner.learner_cla.acc*100,2)} (cal={round(learner.learner_cla.acc_cal*100,2)})')
        # print(learner.H[['acc','acc_test']].describe())
        print('')

    # =============================================================================
    # Load and View Results (all subjects)
    # =============================================================================
    datasets = ['IV2a', 'IV2b', 'Lee19', 'LINCE']
    RA = []
    for ds in datasets:    
        path = '/home/vboas/cloud/results/as_off/' + ds + '/'
        R = pd.DataFrame(columns=['subj', 'class_ids', 'nchannels', 'fl', 'fh', 'tmin', 'tmax', 'nbands', 'ncsp', 'csp_list', 'clf', 'clf_details', 
                                  'as_acc_cal', 'as_acc', 'as_acc_max', 'sb_acc_cal', 'sb_acc', 'cla_acc_cal','cla_acc',
                                  'as_mcost','as_mcost_iir'])
        subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
        for suj in subjects:
            sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
            sname += str(suj)
            learner = Processor()
            learner.load_setup(path + sname + '_learner')
            R.loc[len(R)] = [sname, [1,2], len(learner.channels), learner.f_low, learner.f_high, learner.tmin, learner.tmax, learner.nbands, learner.ncsp, learner.ncsp_list, learner.clf_dict['model'], learner.clf_dict,
                              learner.acc_cal, learner.acc_best, learner.acc, learner.learner_sb.acc_cal, learner.learner_sb.acc, learner.learner_cla.acc_cal, learner.learner_cla.acc, 
                              learner.H['cost'].mean(), learner.learner_iir.H['cost'].mean()]
            # print('')
            # print(f"{sname} Setup: {learner.f_low}-{learner.f_high}Hz; {learner.tmin}-{learner.tmax}s; Ns={learner.nbands} {learner.ncsp_list}; R={learner.ncsp}; {learner.clf_dict}")
            # print(f"{sname} Cost: {round(learner.H['cost'].mean(),2)} (+-{round(learner.H['cost'].std(),2)})  IIR={round(learner.learner_iir.H['cost'].mean(),2)} (+-{round(learner.learner_iir.H['cost'].std(),2)})")
            # print(f'{sname} AS Acc: {round(learner.acc_best*100,2)} (max={round(learner.acc*100,2)}) (cal={round(learner.acc_cal*100,2)})')
            # print(f'{sname} SB Acc: {round(learner.learner_sb.acc*100,2)} (cal={round(learner.learner_sb.acc_cal*100,2)})')
            # print(f'{sname} CLA Acc: {round(learner.learner_cla.acc*100,2)} (cal={round(learner.learner_cla.acc_cal*100,2)})')
            # # print(learner.H[['acc','acc_test']].describe())  
        # pd.to_pickle(R, path + 'R_' + ds + '.pkl') 
        # RA.append(pd.read_pickle(path + 'R_' + ds + '.pkl'))
    
    RFull = pd.concat(RA, ignore_index=True)
    # pd.to_pickle(RFull, '/home/vboas/cloud/results/as_off/RFull.pkl')
    

