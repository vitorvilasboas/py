"""
-*- coding: utf-8 -*-
Created on Sat Mar 14 18:08:50 2020
@author: Vitor Vilas-Boas
"""
import os
import numpy as np
import pandas as pd
from time import time
from scripts.bci_cp import Processor 

if __name__ == "__main__": 
    ds = 'IV2a' # III3a, III4a, IV2a, IV2b, LINCE, Lee19
    overlap = True
    crossval = False
    nfolds = 10 
    test_perc = 0.2 if crossval else 0.5
    cortex_only = True # used when ds == Lee19 - True to used only cortex channels
    
    fl, fh, ncsp, tmin, tmax = 13, 29, 6, 1, 4.
    ncsp_list = None # [6, 6, 2, 6] 
    
    # clf = {'model':'LR'}
    # clf = {'model':'Bayes'}
    # clf = {'model':'LDA', 'lda_solver':'svd'} # 'lda_solver': 'svd','lsqr','eigen'
    clf = {'model':'KNN', 'metric':'manhattan', 'neig':7} # 'metric': 'euclidean','manhattan','minkowski','chebyshev'
    # clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4} # 'kernel': 'linear', 'poly', 'sigmoid', 'rbf'
    # clf = {'model':'MLP', 'eta':-4, 'activ':{'af':'tanh'}, 'alpha':-1, 'n_neurons':465, 'n_hidden':2, 'mlp_solver':'adam'} # 'mlp_solver':'adam', 'lbfgs', 'sgd' # 'af':'identity', 'logistic', 'tanh', 'relu'
    # clf = {'model':'DTree', 'crit':'gini'} # 'crit': 'entropy' or 'gini'
    
    # approach = {'option':'classic'}
    approach = {'option':'sbcsp', 'nbands':2}
    
    filtering = {'design':'DFT'}
    # filtering = {'design':'IIR', 'iir_order':5}
    # filtering = {'design':'FIR', 'fir_order':5} 
    
    if approach['option'] == 'classic': ncsp_list = None
    
    prefix, suffix = '', ''
    if ds == 'III3a':
        subjects = ['K3','K6','L1'] 
        classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]  
    elif ds == 'III4a':
        subjects = ['aa','al','av','aw','ay']
        classes = [[1, 3]]
    elif ds == 'IV2a':        
        subjects = range(1,10) 
        classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        prefix = 'A0'
    elif ds == 'IV2b': 
        subjects = range(1,10)
        classes = [[1, 2]]
        prefix = 'B0'
    elif ds == 'LINCE':
        subjects = ['CL_LR','CL_LF','TL_S1','TL_S2','WL_S1','WL_S2']
        classes = [[1, 2]] 
    elif ds == 'Lee19':
        subjects = range(1, 55) 
        classes = [[1, 2]]
        prefix = 'S'
        suffix = '' # 'sess1' or 'sess2'    
        
    subjects =  [2] # uncomment to run one subject only
    classes = [[1, 2]] # uncomment to run LH x RH classification only
    
    R = pd.DataFrame(columns=['suj','acc']) # ,'kpa','cost'
    for suj in subjects:
        sname = prefix + str(suj) + suffix
        # data, events, info = labeling(path='/mnt/dados/eeg_data/+ds+'/', ds=ds, session='T', subj=suj, channels=None, save=False)
        
        eeg_path_train = '/mnt/dados/eeg_data/' + ds + '/npy/' + sname + 'T.npy'
        eeg_path_test = '/mnt/dados/eeg_data/' + ds + '/npy/' + sname + 'E.npy'
        
        # if ds=='LINCE' and suj == 'CL_LF': classes = [[1, 3]]
        
        cortex54 = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
        
        for class_ids in classes: 
            Fs = 250
            forder = None if filtering['design'] == 'DFT' else filtering['iir_order'] if filtering['design'] == 'IIR' else filtering['fir_order']
            bci = Processor()
        
            if ds=='Lee19' and cortex_only:
                bci.load_eeg_train(eeg_path=eeg_path_train, channels=cortex54)
                bci.load_eeg_test(eeg_path=eeg_path_test, channels=cortex54)
            else:
                bci.load_eeg_train(eeg_path=eeg_path_train, channels=range(0,22))
                bci.load_eeg_test(eeg_path=eeg_path_test, channels=range(0,22))
            
            bci.define_params(f_low=fl, f_high=fh, ncsp=ncsp, class_ids=class_ids, tmin=tmin, tmax=tmax, fs=Fs, 
                              filt_type=filtering['design'], filt_order=forder, clf_dict=clf, 
                              is_sbcsp=True if approach['option']=='sbcsp' else False, 
                              nbands=None if approach['option']=='classic' else approach['nbands'], 
                              overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc, ncsp_list=ncsp_list)
            st = time()
            bci.process(single_eeg=False)
            cost = time() - st
            # print(suj, class_ids, str(round(bci.acc*100,2))+'%', str(round(bci.kpa,3)), str(round(cost, 2))+'s')
            R.loc[len(R)] = [suj, bci.acc] # , bci.kpa, cost
            
    print(R['acc'].mean())

    # pd.to_pickle(R, '/home/vboas/Desktop/results/R54_cla.pkl')
