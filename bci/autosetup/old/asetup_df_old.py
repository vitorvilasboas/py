# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 10:45:20 2020
@author: Vitor Vilas-Boas
"""
import pickle
import numpy as np
import pandas as pd
from time import time
from bci_utils import BCI 

ds = 'Lee19' # III3a, III4a, IV2a, IV2b, Lee19, CL, TW
path_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + 'S' if ds=='Lee19' else 'A0' if ds=='IV2a' else 'B0' if ds=='IV2b' else ''
path_results = '/home/vboas/cloud/devto/overmind/tests/as_results/' + ds + '/'
path_out_df = '/home/vboas/cloud/devto/overmind/tests/as_results/RES_' + ds + '.pickle'

fs = 250 if ds=='Lee19' else 100 if ds=='III4a' else 125 if ds=='CL' else 250
subjects = range(1,55) if ds=='Lee19' else ['aa','al','av','aw','ay'] if ds=='III4a' else ['K3','K6','L1'] if ds=='III4a' else range(1,10)
classes = [[1,3]] if ds=='III4a' else [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]] if ds=='IV2a' else [[1,2]]

only_cortex = True
one_session = True

crossval = False
overlap = True
nfolds = 10
test_perc = 0.1 if crossval else 0.5

print('creating dataframe...')
setup_details = []
for class_ids in classes:
    for suj in subjects:    
        trials = pickle.load(open(path_results + ds + '_' + str(suj) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl', 'rb'))
        
        # print(path_results + ds + '_' + str(suj) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]))
        
        acc = trials.best_trial['result']['loss'] * -1
        
        best = trials.best_trial['misc']['vals']
                 
        fl = int(best['fl'][0])
        fh = int(best['fh'][0])                       
        ncsp = int(best['ncomp'][0])
        approach = 'classic' if best['approach'][0]==0 else 'sbcsp'
        nbands = int(best['nbands'][0]) if best['approach'][0]==1 else 1
        tmin = best['tmin'][0]
        tmax = best['tmax'][0]
        filt = 'DFT' if best['filt'][0]==0 else 'IIR {}'.format(int(best['iir_order'][0]))
        
        while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg
        if nbands > (fh-fl): nbands = (fh-fl)
                  
        clf_type = ''
        clf_details = ''
        
        lda_solver = None
        knn_metric = None
        knn_neig = None
        svm_kernel = None
        svm_clog = None
        mlp_n_hidden = None
        mlp_n_neurons = None
        mlp_eta = None
        mlp_af = None
        mlp_solver = None
        dtree_crit = None
    
        if best['clf'][0] == 0: 
            clf_type += 'Bayes'
        
        elif best['clf'][0] == 1: 
            clf_type += 'LDA'
            lda_solver = 'svd' if best['lda_solver'][0] == 0 else 'lsqr' if best['lda_solver'][0] == 1 else 'eigen'
            clf_details += "solver={}".format(lda_solver)
        
        elif best['clf'][0] == 2: 
            clf_type += 'KNN'
            knn_metric = 'euclidean' if best['metric'][0] == 0 else 'manhattan' if best['metric'][0] == 1 else 'minkowski' if best['metric'][0] == 2 else 'chebyshev'
            knn_neig = int(best['neig'][0])
            clf_details += 'neig={}, metric={}'.format(knn_neig, knn_metric)
               
        elif best['clf'][0] == 3: 
            clf_type += 'SVM'
            svm_kernel = 'linear' if best['kernel'][0]==0 else 'poly' if best['kernel'][0]==1 else 'sigmoid' if best['kernel'][0]==2  else 'rbf'
            svm_clog = int(best['C'][0])
            clf_details += 'k={}, C=10**({})'.format(svm_kernel, svm_clog)
        
        elif best['clf'][0] == 4:
            clf_type += 'MLP'
            mlp_af = 'identity' if best['activ'][0]==0 else 'logistic' if best['activ'][0]==1 else 'tanh' if best['activ'][0]==2  else 'relu'
            mlp_solver = 'adam' if best['mlp_solver'][0] == 0 else 'lbfgs' if best['mlp_solver'][0] == 1 else 'sgd'
            mlp_n_hidden = int(best['n_hidden'][0])
            mlp_n_neurons = int(best['n_neurons'][0])
            mlp_eta = best['eta'][0]
            clf_details += '({}, {}), af={}, eta={}, solver={}'.format(mlp_n_neurons, mlp_n_hidden, mlp_af, mlp_eta, mlp_solver)
        
        elif best['clf'][0] == 5:
            clf_type += 'DTree'
            dtree_crit = 'gini' if best['crit'][0]==0 else 'entropy'
            clf_details += 'criterion={}'.format(dtree_crit)
        
        setup_details.append([suj, '{} {}'.format(class_ids[0], class_ids[1]), acc, fl, fh, filt, ncsp, approach, nbands, tmin, tmax, clf_type, clf_details,
                              lda_solver , knn_metric, knn_neig, svm_kernel, svm_clog, mlp_n_hidden, mlp_n_neurons, mlp_eta,
                              mlp_af, mlp_solver, dtree_crit])
    
    
header = ['subj', 'classes', 'acc', 'fl', 'fh', 'filt', 'ncsp', 'approach', 'nbands', 'tmin', 'tmax', 'clf', 'clf details', 
          'lda_solver', 'knn_metric', 'knn_neig', 'svm_kernel', 'svm_clog', 'mlp_n_hidden', 'mlp_n_neurons', 'mlp_eta', 
          'mlp_af', 'mlp_solver', 'dtree_crit']

S = pd.DataFrame(np.asarray(setup_details), columns=header)


print('calculating kappa and cost...')
RES = []
for class_ids in classes:
    for suj, i in zip(subjects, range(len(subjects))):
        f_low, f_high = S.iloc[i]['fl'], S.iloc[i]['fh']
        tmin, tmax = S.iloc[i]['tmin'], S.iloc[i]['tmax']
        ncomp = S.iloc[i]['ncsp'] 
        approach = {'option': S.iloc[i]['approach'], 'nbands': S.iloc[i]['nbands']}
        filtering = {'design': S.iloc[i]['filt']}       
        clf = {'model': str(S.iloc[i]['clf']), 
               'lda_solver': str(S.iloc[i]['lda_solver']),
               'metric': str(S.iloc[i]['knn_metric']),
               'neig': S.iloc[i]['knn_neig'],
               'kernel': {'kf': str(S.iloc[i]['svm_kernel'])},
               'C': S.iloc[i]['svm_clog'],
               'crit' : str(S.iloc[i]['dtree_crit']),
               'eta': S.iloc[i]['mlp_eta'],
               'activ': {'af': str(S.iloc[i]['mlp_af'])},
               'n_neurons': S.iloc[i]['mlp_n_neurons'],
               'n_hidden': S.iloc[i]['mlp_n_hidden'],
               'mlp_solver': str(S.iloc[i]['mlp_solver'])
               }
        # data, events, info = pickle.load(open(path_data + str(suj) + '.omi', 'rb'))
        data, events, info = np.load(path_data + str(suj) + '.npy', allow_pickle=True)
        
        if ds=='Lee19': 
            if only_cortex:
                
                cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
                data = data[cortex]
                info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
                info['eeg_channels'] = len(cortex)
                print(data.shape, clf, filtering, tmin, tmax, approach, ncomp, f_low, f_high)

        bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, f_low, f_high, tmin, tmax, ncomp, approach, filtering, clf)          
        st = time()
        bci.evaluate()
        cost = time() - st
        RES.append([bci.acc, bci.kappa, round(cost, 4)])    
RES = pd.DataFrame(RES, columns=['acc','kappa', 'cost'])
S.insert(3, 'acc2', RES['acc'])
S.insert(4, 'kappa', RES['kappa'])
S.insert(5, 'cost', RES['cost'])


print('calculating fixed scenario: CSP-LDA + DFT...')
RES = []
for class_ids in classes: # CSP-LDA + DFT
    for suj, i in zip(subjects, range(len(subjects))):
        tmin, tmax = S.iloc[i]['tmin'], S.iloc[i]['tmax']
        f_low, f_high, ncomp = 8, 30, 6
        approach = {'option':'classic'} 
        clf = {'model':'LDA', 'lda_solver':'svd'}
        filtering = {'design':'DFT'}
        # data, events, info = pickle.load(open(path_data + str(suj) + '.omi', 'rb'))
        data, events, info = np.load(path_data + str(suj) + '.npy', allow_pickle=True)
        
        if ds=='Lee19': 
            if only_cortex:
                cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
                data = data[cortex]
                info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
                info['eeg_channels'] = len(cortex)
                
        bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, f_low, f_high, tmin, tmax, ncomp, approach, filtering, clf)          
        st = time()
        bci.evaluate()
        cost = time() - st
        RES.append([bci.acc, bci.kappa, round(cost, 4)]) 
RES = pd.DataFrame(RES, columns=['acc','kappa', 'cost'])
S.insert(16, 'classic_dft_acc', RES['acc'])
S.insert(17, 'classic_dft_kappa', RES['kappa'])
S.insert(18, 'classic_dft_cost', RES['cost'])    


print('calculating fixed scenario: CSP-LDA + IIR...')
RES = []
for class_ids in classes: # CSP-LDA + IIR
    for suj, i in zip(subjects, range(len(subjects))):
        tmin, tmax = S.iloc[i]['tmin'], S.iloc[i]['tmax']
        f_low, f_high, ncomp = 8, 30, 6
        approach = {'option':'classic'} 
        clf = {'model':'LDA', 'lda_solver':'svd'}
        filtering = {'design':'IIR', 'iir_order':5}
        # data, events, info = pickle.load(open(path_data + str(suj) + '.omi', 'rb'))
        data, events, info = np.load(path_data + str(suj) + '.npy', allow_pickle=True)
                
        if ds=='Lee19': 
            if only_cortex:
                cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
                data = data[cortex]
                info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
                info['eeg_channels'] = len(cortex)
                
        bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, f_low, f_high, tmin, tmax, ncomp, approach, filtering, clf)          
        st = time()
        bci.evaluate()
        cost = time() - st
        RES.append([bci.acc, bci.kappa, round(cost, 4)]) 
RES = pd.DataFrame(RES, columns=['acc','kappa', 'cost'])
S.insert(19, 'classic_iir_acc', RES['acc'])
S.insert(20, 'classic_iir_kappa', RES['kappa'])
S.insert(21, 'classic_iir_cost', RES['cost']) 


print('calculating fixed scenario: SBCSP + DFT...')
RES = []
for class_ids in classes: # SBCSP + DFT
    for suj, i in zip(subjects, range(len(subjects))):
        tmin, tmax = S.iloc[i]['tmin'], S.iloc[i]['tmax']
        f_low, f_high, ncomp = 0, 40, 6
        approach = {'option':'sbcsp', 'nbands':10} 
        clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4}
        filtering = {'design':'DFT'}
        # data, events, info = pickle.load(open(path_data + str(suj) + '.omi', 'rb'))
        data, events, info = np.load(path_data + str(suj) + '.npy', allow_pickle=True) 
                
        if ds=='Lee19': 
            if only_cortex:
                cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
                data = data[cortex]
                info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
                info['eeg_channels'] = len(cortex)
                
        bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, f_low, f_high, tmin, tmax, ncomp, approach, filtering, clf)          
        st = time()
        bci.evaluate()
        cost = time() - st
        RES.append([bci.acc, bci.kappa, round(cost, 4)]) 
RES = pd.DataFrame(RES, columns=['acc','kappa', 'cost'])
S.insert(22, 'sb_dft_acc', RES['acc'])
S.insert(23, 'sb_dft_kappa', RES['kappa'])
S.insert(24, 'sb_dft_cost', RES['cost']) 


print('calculating fixed scenario: SBCSP + IIR...')
RES = []
for class_ids in classes: # SBCSP + IIR
    for suj, i in zip(subjects, range(len(subjects))):
        tmin, tmax = S.iloc[i]['tmin'], S.iloc[i]['tmax']
        f_low, f_high, ncomp = 0, 40, 6
        approach = {'option':'sbcsp', 'nbands':10} 
        clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4}
        filtering = {'design':'IIR', 'iir_order':5}
        # data, events, info = pickle.load(open(path_data + str(suj) + '.omi', 'rb'))
        data, events, info = np.load(path_data + str(suj) + '.npy', allow_pickle=True)
                
        if ds=='Lee19': 
            if only_cortex:
                cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
                data = data[cortex]
                info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
                info['eeg_channels'] = len(cortex)
                
        bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, f_low, f_high, tmin, tmax, ncomp, approach, filtering, clf)          
        st = time()
        bci.evaluate()
        cost = time() - st
        RES.append([bci.acc,bci.kappa, round(cost, 4)]) 
RES = pd.DataFrame(RES, columns=['acc','kappa', 'cost'])
S.insert(25, 'sb_iir_acc', RES['acc'])
S.insert(26, 'sb_iir_kappa', RES['kappa'])
S.insert(27, 'sb_iir_cost', RES['cost']) 

pd.to_pickle(S, path_out_df)




