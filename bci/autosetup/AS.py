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
from scripts.ASS_CP import Processor
from scripts.ASS_AS import AutoSetup

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

if __name__ == "__main__":  
    n_iter = 200
    class_ids = [1,2]
    ds = 'IV2a' # IV2a, IV2b, Lee19, LINCE
    
    # test_id = datetime.now().today().strftime("%m-%d %H-%M-%S")
    # path = '/home/vboas/cloud/results/as_off/' + ds + '_' + test_id + '/'
    path = '/home/vboas/cloud/results/as_off/' + ds + '/'
    if not os.path.isdir(path): os.makedirs(path)
    
    subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
    subjects = [2]
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
        
        asetup = AutoSetup(setup=bci, n_iter=n_iter, load_last_setup=False, dump=True, 
                           path_dump='/home/vboas/Desktop/TESTES/'+sname+'_trials.pkl')
        t0 = time()
        asetup.run_optimizer()
        cost = time() - t0
        H = asetup.H

        # acc_teste = []
        # for i in range(len(H)):
        #     bci = copy.copy(H.iloc[i]['learner'])
        #     bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        #     bci.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
        #     # bci.ncsp_list = learner_max.ncsp_list
        #     bci.crossval = False
        #     try: 
        #         bci.process()
        #         acc_teste.append(bci.acc)
        #     except: acc_teste.append(0.1)
        #     t = bci.t
        #     V.append(bci.y)
        #     P.append(bci.y_prob)
        # H.insert(loc=8, column='acc_test', value=acc_teste)
        
        ### Auto Setup Val
        acc_cal = H[ H['acc'] == H['acc'].max()].iloc[0]['acc']
        hmax = H[ H['acc'] == H['acc'].max()].iloc[0] # acc_test
        bci = copy.copy(hmax['learner'])
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        bci.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
        bci.filt_type, bci.filt_order = 'DFT', None
        bci.crossval = False
        bci.process()
        learner = bci

        learner.load_eeg_train(eeg_path=eeg_path_train, channels=ch)
        learner.load_eeg_test(eeg_path=eeg_path_test, channels=ch)
        learner.crossval = False
        learner.process()
        learner.clear_eeg_data()
        
        learner.cost = cost
        learner.acc_cal = acc_cal # hmax.acc
        
        as_trials = asetup.trials
        print("")
        print(f"{sname} val={round(learner.acc*100,2)}, cal={round(acc_cal*100,2)}, cost={round(H['cost'].mean(),2)}, setup=[{learner.f_low}-{learner.f_high}Hz; {learner.tmin}-{learner.tmax}s; Ns={learner.nbands} {learner.ncsp_list}; R={learner.ncsp}; {learner.clf_dict}]")
        # learner.save_setup('/home/vboas/Desktop/' + sname + '_learner')

    # # =============================================================================
    # # Load and View Results (all subjects)
    # # =============================================================================
    # datasets = ['IV2a', 'IV2b', 'Lee19', 'LINCE']
    # RA = []
    # for ds in datasets:    
    #     path = '/home/vboas/cloud/results/as_off/' + ds + '/'
    #     R = pd.DataFrame(columns=['subj', 'class_ids', 'nchannels', 'fl', 'fh', 'tmin', 'tmax', 'nbands', 'ncsp', 'csp_list', 'clf', 'clf_details', 
    #                               'as_acc_cal', 'as_acc', 'as_acc_max', 'sb_acc_cal', 'sb_acc', 'cla_acc_cal','cla_acc',
    #                               'as_mcost','as_mcost_iir'])
    #     subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
    #     for suj in subjects:
    #         sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
    #         sname += str(suj)
    #         learner = Processor()
    #         learner.load_setup(path + sname + '_learner')
    #         R.loc[len(R)] = [sname, [1,2], len(learner.channels), learner.f_low, learner.f_high, learner.tmin, learner.tmax, learner.nbands, learner.ncsp, learner.ncsp_list, learner.clf_dict['model'], learner.clf_dict,
    #                           learner.acc_cal, learner.acc_best, learner.acc, learner.learner_sb.acc_cal, learner.learner_sb.acc, learner.learner_cla.acc_cal, learner.learner_cla.acc, 
    #                           learner.H['cost'].mean(), learner.learner_iir.H['cost'].mean()]
    #         # print('')
    #         # print(f"{sname} Setup: {learner.f_low}-{learner.f_high}Hz; {learner.tmin}-{learner.tmax}s; Ns={learner.nbands} {learner.ncsp_list}; R={learner.ncsp}; {learner.clf_dict}")
    #         # print(f"{sname} Cost: {round(learner.H['cost'].mean(),2)} (+-{round(learner.H['cost'].std(),2)})  IIR={round(learner.learner_iir.H['cost'].mean(),2)} (+-{round(learner.learner_iir.H['cost'].std(),2)})")
    #         # print(f'{sname} AS Acc: {round(learner.acc_best*100,2)} (max={round(learner.acc*100,2)}) (cal={round(learner.acc_cal*100,2)})')
    #         # print(f'{sname} SB Acc: {round(learner.learner_sb.acc*100,2)} (cal={round(learner.learner_sb.acc_cal*100,2)})')
    #         # print(f'{sname} CLA Acc: {round(learner.learner_cla.acc*100,2)} (cal={round(learner.learner_cla.acc_cal*100,2)})')
    #         # # print(learner.H[['acc','acc_test']].describe())  
    #     pd.to_pickle(R, path + 'R_' + ds + '.pkl') 
    #     RA.append(pd.read_pickle(path + 'R_' + ds + '.pkl'))
    
    # RFull = pd.concat(RA, ignore_index=True)
    # pd.to_pickle(RFull, '/home/vboas/cloud/results/as_off/RFull.pkl')