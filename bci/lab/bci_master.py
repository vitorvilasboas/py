# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import time
from hyperopt import base, fmin, tpe, hp
from bci_utils import BCI, iii3a_save_npy, iii4a_save_npy, iv2a_save_npy, iv2b_save_npy, cl_save_npy, twl_save_npy
from tqdm import tqdm
#pbar = tqdm(total=max_evals, desc="Hyperopt")

if __name__ == '__main__':

    dataset = 'BCI3_4a' # BCI3_3a, BCI3_4a, BCI4_2a, BCI4_2b, CL, TWL
    npy_create = False  # True to create npy files
    auto_setup = False
    crossval = False
    overlap = True
    nfolds = 10
    test_perc = 0.2
    path = '/mnt/dados/eeg_data/' + dataset

    
    if dataset == 'BCI3_3a':
        """ 3 sujeitos (K3, K6, L1) | 4 classes | 60 canais | Fs 250Hz
            K3->(360 trials (90 por classe)) - 2 sessões
            K6,L1->(240 trials (60 por classe)) - 2 sessões 
            startTrial=0; beep/cross=2; startCue=3; startMI=4; endMI=7; endTrial(break)=10    
        """
        all_subjects = ['K3','K6','L1'] # 
        all_classes = [1, 2, 3, 4]
        pair_classes = [[1, 2]] #[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        n_eeg_channels = 60
        max_ncomp = n_eeg_channels/2 if all_subjects == ['K6'] else n_eeg_channels
        startMI, endMI = 3, 7
        trials_per_class = 90 if all_subjects == ['K3'] else 60 # K3=90, K6=60, L1=60
        max_nbands = 48
        fs = 250.0
        if npy_create: [ iii3a_save_npy(suj, path) for suj in all_subjects ]
        path += '/omi/' # '/npy/' or '/omi/'

    
    elif dataset == 'BCI3_4a':
        """ 5 subjects | 2 classes (RH, FooT)
            Epoch distribution:
                aa : train=168 test=112  
                al : train=224 test=56
                av : train=84  test=196
                aw : train=56  test=224
                ay : train=28  test=252
        """
        all_subjects = ['aa'] # ['aa','al','av','aw','ay']
        all_classes = [1, 3]
        pair_classes = [[1, 3]]
        n_eeg_channels = 118
        max_ncomp = n_eeg_channels/2
        startMI, endMI = 0, 4.5
        trials_per_class = 140
        max_nbands = 40
        fs = 100.0
        if npy_create: [ iii4a_save_npy(suj, path) for suj in all_subjects ]
        path += '/omi/' # '/npy/' or '/omi/'

        
    elif dataset == 'BCI4_2a':        
        """ 72 trials per classe * 2 sessions
            T = startTrial=0; cue=2; startMI=3.25; endMI=6; endTrial=7.5-8.5
        """
        all_subjects = range(1,10) # [1] #
        all_classes = [1, 2, 3, 4]
        pair_classes = [[1, 2]] # [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] # [[1, 2]] # 
        n_eeg_channels = 22
        max_ncomp = n_eeg_channels
        startMI, endMI = 2, 6
        trials_per_class = 144
        max_nbands = 50
        fs = 250.0
        if npy_create: [ iv2a_save_npy(suj, path, channels=n_eeg_channels) for suj in all_subjects ]
        path += '/npy/A0' # '/npy/A0' or '/omi/A0'


    elif dataset == 'BCI4_2b':
        """ 9 subjects | 2 classes (LH, RH) | 3 channels | Fs 250Hz
            6 channels (first 3 is EEG: C3, C4, Cz; last 3 is EOG)
            120 trials (60 per class) - 5 sessions
            2 sessions without feedback
            3 sessions with feedback (smiley)
            Total sessions = 5 (01T,02T,03T,04E,05E)
                 5 * 120 trials = 600 total trials -> 5*60 = 300 per class -> 2*60 = 120 per session
        	     2 training sessions (no feedback) - 01T,02T 
                 1 training session (WITH feedback) - 03T
        	     2 evaluate sessions (WITH feedback) - 04E,05E
                 
            # startTrial=0; cue=3; startMI=4; endMI=7; endTrial=8.5-9.5
        """
        all_subjects = range(1,10)
        all_classes = [1, 2]
        pair_classes = [[1, 2]]
        n_eeg_channels = 3
        max_ncomp = n_eeg_channels 
        startMI, endMI = 3, 7
        trials_per_class = 120 # total = 360
        max_nbands = 48
        fs = 250.0
        if npy_create: [ iv2b_save_npy(suj, path) for suj in all_subjects ]
        path += '/omi/B0' # '/npy/B0' or '/omi/B0'

  
    elif dataset == 'CL':
        """ 1 subject (CL) | 3 classes (lh, rh, foot) | 16 channels | Fs 125Hz
            lh-rh -> 100 trials (50 per class) 5*20 - 1 session
            lh-ft -> 48 trials (24 per class) 3*16 - 1 session
            Start trial=0; Beep=1; Wait=2; Start cue=2; Start MI=3; End MI=9; End trial(break)=14
        """
        all_subjects = ['CL_LF'] #['CL_LR', 'CL_LF'] # 
        all_classes = [1, 2, 3]
        pair_classes = [[1, 2]]
        n_eeg_channels = 16
        max_ncomp = n_eeg_channels - 4
        startMI, endMI = 2, 9
        trials_per_class = 50 if all_subjects == ['CL_LR'] else 24 # LR=50, LF=24
        max_nbands = 50
        fs = 125.0
        if npy_create: [ cl_save_npy(suj, path) for suj in all_subjects ]
        path += '/omi/' # '/npy/' or '/omi/'

        
    elif dataset == 'TWL':
        """ 2 subjects (TL, WL) | 2 classes (lh, rh) | Fs 250Hz
            40 trials (20 per class) - TL: 2 sessions; WL:3 sessions
            8 channels (1=Cz 2=Cpz 3=C1 4=C3 5=CP3 6=C2 7=C4 8=CP4)
            Scalp map:      C3  C1  Cz  C2  CP4     4  3  1  6  7
                               CP3   CPz  CP4 		  5   2   8
            Start trial=0; Wait beep=2; Start cue=3; Start MI=4.25; End MI=8; End trial(break)=10-12
        """
        all_subjects = ['WL'] #['TL', 'WL'] 
        all_classes = [1, 2]
        pair_classes = [[1, 2]]
        n_eeg_channels = 8
        max_ncomp = n_eeg_channels - 2
        startMI, endMI = 3, 8
        trials_per_class = 40
        max_nbands = 50
        fs = 250.0
        if npy_create: [ twl_save_npy(suj, path) for suj in all_subjects ]
        path += '/omi/' # '/npy/' or '/omi/'

    
    if auto_setup:
        
        n_iter = 100
        space = (
            hp.uniformint('fl', 0, 20), # hp.quniform('fl', 1, 20, 1),
            hp.uniformint('fh', 30, 49),  # hp.quniform('fh', 30, 49, 1),
            hp.quniform('tmin', startMI, startMI+2, 0.5),
            hp.quniform('tmax', startMI+2, endMI, 0.5),
            hp.quniform('ncomp', 2, max_ncomp, 2),
            hp.choice('approach', [
                {'option':'classic',},
                {'option':'sbcsp', 'nbands': hp.uniformint('nbands', 2, max_nbands)} # hp.quniform('nbands', 2, max_nbands, 1)}
                ]),
            hp.choice('filt', [
                {'design':'DFT'},
                # {'design':'IIR', 'iir_order': hp.uniformint('iir_order', 1, 8)}, #hp.quniform('iir_order', 1, 8, 1)},
                #{'design':'FIR', 'fir_order': hp.uniformint('fir_order', 1, 8)}, #hp.quniform('fir_order', 1, 8, 1)}
                ]),
            hp.choice('clf', [
                {'model':'Bayes'},
                {'model':'LDA',
                  'lda_solver': hp.choice('lda_solver', ['svd','lsqr','eigen']),
                  # 'shrinkage': hp.choice('shrinkage', [None, 'auto', {'shrinkage_float':  hp.uniform('shrinkage_float', 0, 1)}]) #np.logspace(-4, 0, 1)
                  },
                {'model':'KNN', 
                  'neig': hp.uniformint('neig', 2, trials_per_class), # hp.quniform('neig', 2, trials_per_class, 1)
                  'metric': hp.choice('metric', ['euclidean','manhattan','minkowski','chebyshev']), #{'mf':'cityblock'}, {'mf':'cosine'}, {'mf':'l1'}, {'mf':'l2'},
                  # 'p': hp.quniform('p', 2, 50, 1)
                  },
                {'model':'SVM', 
                  'C': hp.quniform('C', -8, 4, 1), 
                  'kernel': hp.choice('kernel', [{'kf':'linear'}, {'kf':'poly',}, {'kf':'sigmoid'}, {'kf':'rbf'}]), #'degree': hp.uniformint('degree', 2, 4)    #, 'width': hp.lognormal('width', 0, 1)
                  # 'gamma': hp.choice('gamma', ['scale', 'auto', {'gamma_float': hp.quniform('gamma_float', -9, 4, 1)}]), # hp.loguniform('gamma_float', -9, 3)  np.logspace(-9, 3, 13)),
                  },
                {'model':'MLP', 
                  'eta': hp.quniform('eta', -8, 1, 1), 
                  'alpha': hp.quniform('alpha', -8, 1, 1), 
                  'n_neurons' : hp.uniformint('n_neurons', 50, 500), # hp.quniform('n_neurons', 50, 500, 50), 
                  'n_hidden': hp.uniformint('n_hidden', 1, 4), # hp.quniform('n_hidden', 1, 4, 1),
                  'activ': hp.choice('activ', [{'af':'identity'},{'af':'logistic'},{'af':'tanh'},{'af':'relu'}]),
                  # 'eta_type': hp.choice('eta_type', ['constant', 'invscaling', 'adaptive']),
                  'mlp_solver': hp.choice('mlp_solver', ['adam', 'lbfgs', 'sgd']),
                  },
                {'model':'DTree', 
                  'crit': hp.choice('crit', ['gini', 'entropy']),
                  # 'max_depth': hp.choice('max_depth', [None, {'max_depth_int': hp.qlognormal('max_depth_int', 3, 1, 1)}]), # np.random.lognormal(3, 1, 1) ]),
                  # 'min_split': hp.uniform('min_split', 0.0001, 1), #  np.random.lognormal(2, 1, 1) # hp.qlognormal('min_split', 2, 1, 1)
                  }
                ])
            )
        
        ALL_SETUPS = []
        RES = []
        for suj in all_subjects:
            # data = np.load(path + str(suj) + '_data.npy')[:n_eeg_channels]
            # events = np.load(path + str(suj) + '_events.npy')
            data, events, info = pickle.load(open(path + str(suj) + '.omi', 'rb'))
            
            for class_ids in pair_classes:
                
                path_to_setup = '/mnt/dados/eeg_data/_setup/' + dataset + '_s' + str(suj) + '_cl' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pickle'
                
                bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc)
                
                trials = base.Trials()  
                # try:
                #     print('Trying to pickle file')
                #     trials = pickle.load(open(path_to_setup, 'rb'))
                # except:
                #     print('No trial file at specified path, creating new one')
                #     trials = base.Trials()
                # else:
                #     print('File found')
                
                try:
                    print('Size of object: ' + str(len(trials)))
                    st = time()
                    best = fmin(bci.objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=1)
                    # pickle.dump(trials, open(path_to_setup, 'wb'))
                    print(suj, class_ids, best)
                except:
                    print('Exception raised')
                    # pickle.dump(trials, open(path_to_setup, 'wb'))
                    # print('\n', trials.best_trial['misc']['vals'])
                    raise
                
                ALL_SETUPS.append([suj, class_ids, bci, trials.best_trial])
                
                acc = trials.best_trial['result']['loss']

                # print('\n### BEST SETUP ###')
                # print(f"{suj} {class_ids} {round(trials.best_trial['result']['loss']*-100, 2)}% ")
                # print('{}-{}Hz  {}csp'.format(int(best['fl']), int(best['fh']), int(best['ncomp'])))
                # print('{}'.format('single' if best['approach']==0 else 'SB'))
                # print('{}sb'.format(int(best['nbands']) if best['approach']==1 else ' -'))
                # print('{}-{}s'.format(best['tmin'],best['tmax']))
                # # print('{}'.format('DFT' if best['filt']==0 else 'IIR'))
                # # if best['filt'] in [1,2]: print('{}'.format(int(best['iir_order'])))
                
                # if best['clf'] == 0: print('Bayes')
                # elif best['clf'] == 1: 
                #     lda_solver = 'svd' if best['lda_solver'] == 0 else 'lsqr' if best['lda_solver'] == 1 else 'eigen'
                #     print(f"LDA solver={'lda_solver'}")
                # elif best['clf'] == 2: 
                #     if best['metric'] == 0: mf = 'euclidean'
                #     if best['metric'] == 1: mf = 'manhattan'
                #     if best['metric'] == 2: mf = 'minkowski'
                #     if best['metric'] == 3: mf = 'chebyshev'
                #     print(f"KNN neig={int(best['neig'])}, metric={mf}")
                # elif best['clf'] == 3: 
                #     kf = 'linear' if best['kernel']==0 else 'poly' if best['kernel']==1 else 'sigmoid' if best['kernel']==2  else 'rbf'
                #     print(f"SVM k={kf}, C=10**({best['Clog']}), ")
                # elif best['clf'] == 4:
                #     af = 'identity' if best['activ']==0 else 'logistic' if best['activ']==1 else 'tanh' if best['activ']==2  else 'relu'
                #     eta_type = 'adaptive'
                #     if best['mlp_solver'] == 0: mlp_solver = 'adam'
                #     elif best['mlp_solver'] == 1: mlp_solver = 'lbfgs'
                #     elif best['mlp_solver'] == 2: mlp_solver = 'sgd'
                #     print(f"MLP ({int(best['n_neurons'])}, {int(best['n_hidden'])}) af={af}  eta={best['eta']} alpha={best['alpha']} eta_type={eta_type} solver={mlp_solver}")
                # elif best['clf'] == 5:  
                #     crit = 'gini' if best['crit']==0 else 'entropy'
                #     print(f"DTree criterion={crit}")
                
                RES.append([suj, class_ids, acc])
        
        RES = np.array(RES)      
        print(f'Mean Acc: {round(np.mean(RES[:,2])*100,2)}~{round(np.std(RES[:,2])*100,1)}%')
        
    
    else:
        fl, fh, ncsp, tmin, tmax = 4, 36, 2, 0.5, 4.0
        
        # clf = {'model':'LDA', 'lda_solver':'svd'}
        # clf = {'model':'Bayes'}
        # clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4} # 'kernel': 'linear', 'sigmoid', 'poly', 'rbf'
        clf = {'model':'KNN', 'metric':'manhattan', 'neig':111}
        # clf = {'model':'DTree', 'crit':'gini'} # 'crit': 'entropy' or 'gini'
        # clf = {'model':'MLP', 'eta':-2, 'activ':'identity', 'alpha':-2, 'n_neurons':150, 'n_hidden':1}

        # approach = {'option':'classic'}
        approach = {'option':'sbcsp', 'nbands':30}
        
        filtering = {'design':'DFT'}
        # filtering = {'design':'IIR', 'iir_order':5}
        # filtering = {'design':'FIR', 'fir_order':5}
        
        RES = []
        for suj in all_subjects:
            # data = np.load(path + str(suj) + '_data.npy')[:n_eeg_channels]
            # events = np.load(path + str(suj) + '_events.npy')
            data, events, info = pickle.load(open(path + str(suj) + '.omi', 'rb'))
            
            for class_ids in pair_classes:
                bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, fl, fh, tmin, tmax, ncsp, approach, filtering, clf)  
                start = time() 
                acc = bci.evaluate()
                cost = time() - start # cost estimate
                RES.append([suj,class_ids,acc,cost])
                print(suj, class_ids, str(round(acc*100,2))+'%', str(round(cost,3))+'s')
        RES = np.array(RES)  
        print(f'\nMA {str(round(RES[:,2].mean()*100,2))}%  +- {str(round(RES[:,2].std()*100,1))}% || {round(RES[:,3].sum(),3)}s')
        
        
        # RES1 = RES[:,:3]
        # RES1[:,2] = [round(RES[i,2]*100,2) for i in range(len(RES1))]
        # ACC = RES1[:,2].reshape(6,9)
        # [ print(round(ACC[i,:].mean(),2), round(ACC[i,:].std(),1)) for i in range(0,6) ]
        # [ print(round(ACC[:,j].mean(),2), round(ACC[:,j].std(),1)) for j in range(0,9) ]
        
        # plt.figure(figsize=(10, 7), facecolor='mintcream')
        # plt.grid(axis='y', **dict(ls='--', alpha=0.6))
        # plt.boxplot(ACC, vert = True, showfliers = True, notch = False, patch_artist = True, 
        #             boxprops=dict(facecolor="lavender", color="gray", linewidth=1, hatch = ''))
        # plt.xlabel('Subject', size=14)
        # plt.ylabel('Accuracy (%)', size=14)
        # plt.yticks(np.arange(50, 100, step=5))
        # plt.xticks(np.arange(1, 10, step=1))
        # # plt.title('Boxplot: Acurácia do classificador LDA por sujeito (MD x ME) - 28 janelas x 9 sujeitos')
        # plt.savefig('boxplot_subj_SBCSP_IIR_0-40.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
        
        # plt.figure(figsize=(10, 7), facecolor='mintcream')
        # plt.grid(axis='y', **dict(ls='--', alpha=0.6))
        # plt.boxplot(ACC.T, vert = True, showfliers = True, notch = False, patch_artist = True, 
        #             boxprops=dict(facecolor="lavender", color="gray", linewidth=1, hatch = ''))
        # plt.xlabel('Task', size=14)
        # plt.ylabel('Accuracy (%)', size=14)
        # plt.yticks(np.arange(50, 100, step=5))
        # plt.xticks(np.arange(0, 7), ('','LR', 'LF', 'LT', 'RF', 'RT', 'FT'))
        # # plt.title('Boxplot: Acurácia do classificador LDA por sujeito (MD x ME) - 28 janelas x 9 sujeitos')
        # plt.savefig('boxplot_tasks_SBCSP_IIR_0-40.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
                