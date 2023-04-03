# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import math
import pickle
import numpy as np
import pandas as pd
from time import time
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from bci_utils import extractEpochs, nanCleaner, Filter, CSP
from sklearn.metrics import cohen_kappa_score 


ds = 'LEE54' # III3a, III4a, IV2a, IV2b, LEE54, CL, TWL
fs = 250 if ds=='LEE54' else 100 if ds=='III4a' else 125 if ds=='CL' else 250
subjects = range(1,55) if ds=='LEE54' else range(1,10) 
class_ids = [1, 2]
path = '/mnt/dados/eeg_data/' + ds + '/omi_s2_cortex/' + ('subj' if ds=='LEE54' else 'A0')

S = pd.read_pickle('/home/vboas/cloud/auto_setup_results/' + ds + '_s2_cortex/setup_list2.pickle')

crossval = False
overlap = True
nfolds = 10
test_perc = 0.1 if crossval else 0.5

RES_CLASSIC = []
RES_SBCSP = []

for cenario in range(2):
    
    if cenario == 0:
        # Cenário CSP-LDA
        f_low, f_high, ncomp = 8, 30, 6
        clf = {'model':'LDA', 'lda_solver':'svd'}
        approach = {'option':'classic'}
           
    else:
        # Cenário SBCSP
        f_low, f_high, ncomp = 0, 40, 6
        clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4}
        approach = {'option':'sbcsp', 'nbands':10}
    
    for suj in subjects:
        
        for ft in ['DFT', 'IIR']:
        
            tmin, tmax = S.iloc[suj-1]['tmin'], S.iloc[suj-1]['tmax']  
            
            
            filtering = {'design':'DFT'} if ft=='IIR' else {'design':'IIR', 'iir_order':5}
            
            ###################################################
            
            data, events, info = pickle.load(open(path + str(suj) + '.omi', 'rb'))
            
            # print(data.shape, events.shape)
            
            smin = math.floor(tmin * fs)
            smax = math.floor(tmax * fs)
            buffer_len = smax - smin
            
            dft_rf = fs/buffer_len # resolução em frequência fft
            dft_size_band = round(2/dft_rf) # 2 representa sen e cos que foram separados do componente complexo da fft intercalados     
            
            epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
            epochs = nanCleaner(epochs)
            # self.epochs = np.asarray([ nanCleaner(ep) for ep in self.epochs ])
            
            
            filt = Filter(f_low, f_high, buffer_len, fs, filtering)
            
            csp = CSP(n_components=ncomp)
            
            if clf['model'] == 'LDA': clf_final = LDA(solver=clf['lda_solver'], shrinkage=None)
            if clf['model'] == 'Bayes': clf_final = GaussianNB()
            if clf['model'] == 'SVM': clf_final = SVC(kernel=clf['kernel']['kf'], C=10 ** (clf['C']), gamma='scale', degree=3, probability=True)
            if clf['model'] == 'KNN': clf_final = KNeighborsClassifier(n_neighbors=int(clf['neig']), metric=clf['metric'], p=3)           
            if clf['model'] == 'DTree': clf_final = DecisionTreeClassifier(criterion=clf['crit'], random_state=0, max_depth=None, min_samples_split=2)  
            if clf['model'] == 'MLP': clf_final = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4, learning_rate_init=10**clf['eta'], # alpha=10**clf['alpha'], 
                                                                activation=clf['activ']['af'], hidden_layer_sizes=(int(clf['n_neurons']), int(clf['n_hidden'])),
                                                                learning_rate='constant', solver=clf['mlp_solver'])
            
            if crossval:
                
                cross_scores = []
                cross_kappa = []
                
                kf = StratifiedShuffleSplit(nfolds, test_size=test_perc, random_state=42)
                # kf = StratifiedKFold(nfolds, False)
                
                st = time()
                
                if approach['option'] == 'classic':
                    
                    XF = filt.apply_filter(epochs)
                    
                    if filtering['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
                        bmin = f_low * dft_size_band
                        bmax = f_high * dft_size_band
                        XF = XF[:, :, bmin:bmax]
                        # print(bmin, bmax)
                    
                    # chain = Pipeline([('CSP', csp), ('SVC', clf_final)])
                    # cross_scores = cross_val_score(chain, XF, labels, cv=kf)
                    
                    for idx_treino, idx_teste in kf.split(XF, labels):
                        XT, XV, yT, yV = XF[idx_treino], XF[idx_teste], labels[idx_treino], labels[idx_teste]
                        
                        # Option 1
                        csp.fit(XT, yT)
                        XT_CSP = csp.transform(XT)
                        XV_CSP = csp.transform(XV) 
                        clf_final.fit(XT_CSP, yT)
                        scores = clf_final.predict(XV_CSP)
                        csp_filters = csp.filters_
                        
                        # Option 2
                        # chain = Pipeline([('CSP', csp), ('SVC', clf_final)])
                        # chain.fit(XT, yT)
                        # scores = chain.predict(XV)
                        # csp_filters = chain['CSP'].filters_
                        
                        acc_fold = np.mean(scores == yV) # or cross_scores.append(chain.score(XV, yV)) 
                        kappa_fold = cohen_kappa_score(scores, yV)     
                        cross_scores.append(acc_fold)
                        cross_kappa.append(kappa_fold)
                
                
                
                elif approach['option'] == 'sbcsp':
                    
                    for idx_treino, idx_teste in kf.split(epochs, labels):
                        
                        XT, XV, yT, yV = epochs[idx_treino], epochs[idx_teste], labels[idx_treino], labels[idx_teste]
                        
                        
                        nbands = int(approach['nbands'])
                        if nbands > (f_high-f_low): nbands = (f_high-f_low)
                        
                        n_bins = f_high - f_low
                        overlap = 0.5 if overlap else 1
                        step = n_bins / nbands
                        size = step / overlap
                        
                        sub_bands = []
                        for i in range(nbands):
                            fl_sb = round(i * step + f_low)
                            fh_sb = round(i * step + size + f_low)
                            if fh_sb <= f_high: sub_bands.append([fl_sb, fh_sb]) 
                            # se ultrapassar o limite superior da banda total, desconsidera a última sub-banda
                            # ... para casos em que a razão entre a banda total e n_bands não é exata 
                        
                        nbands = len(sub_bands)
                        
                        XTF, XVF = [], []
                        if filtering['design'] == 'DFT':
                            XT_FFT = filt.apply_filter(XT)
                            XV_FFT = filt.apply_filter(XV)
                            for i in range(nbands):
                                bmin = sub_bands[i][0] * dft_size_band
                                bmax = sub_bands[i][1] * dft_size_band
                                XTF.append(XT_FFT[:, :, bmin:bmax])
                                XVF.append(XV_FFT[:, :, bmin:bmax])
                                # print(bmin, bmax)
                                
                        elif filtering['design'] in ['IIR' or 'FIR']:
                            for i in range(nbands):
                                filt_sb = Filter(sub_bands[i][0], sub_bands[i][1], len(XT[0,0,:]), fs, filtering)
                                XTF.append(filt_sb.apply_filter(XT))
                                XVF.append(filt_sb.apply_filter(XV))
                        
                        chain = [ Pipeline([('CSP', CSP(n_components=ncomp)), ('LDA', LDA())]) for i in range(nbands) ]
                        
                        for i in range(nbands): chain[i]['CSP'].fit(XTF[i], yT)
                            
                        XT_CSP = [ chain[i]['CSP'].transform(XTF[i]) for i in range(nbands) ]
                        XV_CSP = [ chain[i]['CSP'].transform(XVF[i]) for i in range(nbands) ]
                        
                        SCORE_T = np.zeros((len(XT), nbands))
                        SCORE_V = np.zeros((len(XV), nbands))
                        for i in range(len(sub_bands)): 
                            chain[i]['LDA'].fit(XT_CSP[i], yT)
                            SCORE_T[:, i] = np.ravel(chain[i]['LDA'].transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
                            SCORE_V[:, i] = np.ravel(chain[i]['LDA'].transform(XV_CSP[i]))
                        
                        csp_filters_sblist = [chain[i]['CSP'].filters_ for i in range(nbands)]
                        lda_sblist = [chain[i]['LDA'] for i in range(nbands)] 
                        
                        SCORE_T0 = SCORE_T[yT == class_ids[0], :]
                        SCORE_T1 = SCORE_T[yT == class_ids[1], :]
                        p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
                        p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
                        META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
                        META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
                
                        clf_final.fit(META_SCORE_T, yT)
                        scores = clf_final.predict(META_SCORE_V)
                                                
                        acc_fold = np.mean(scores == yV) # calculo acurácia
                        kappa_fold = cohen_kappa_score(scores, yV)
                               
                        cross_scores.append(acc_fold)
                        cross_kappa.append( kappa_fold )
                
                
                if filtering['design'] == 'DFT': 
                    acc_dft = np.mean(cross_scores)
                    kappa_dft = np.mean(cross_kappa)
                    cost_dft = time() - st
                
                if filtering['design'] == 'IIR': 
                    acc_iir = np.mean(cross_scores)
                    kappa_iir = np.mean(cross_kappa)
                    cost_iir = time() - st
                
                
                
            else: # not crossval
                
                test_size = int(len(epochs) * test_perc)
                train_size = int(len(epochs) - test_size)
                train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
                epochsT, labelsT = epochs[:train_size], labels[:train_size] 
                epochsV, labelsV = epochs[train_size:], labels[train_size:]
                
                XT = [ epochsT[np.where(labelsT == i)] for i in class_ids ] # Extrair épocas de cada classe
                XV = [ epochsV[np.where(labelsV == i)] for i in class_ids ]
                
                XT = np.concatenate([XT[0],XT[1]]) # Train data classes A + B
                XV = np.concatenate([XV[0],XV[1]]) # Test data classes A + B        
                yT = np.concatenate([class_ids[0] * np.ones(int(len(XT)/2)), class_ids[1] * np.ones(int(len(XT)/2))])
                yV = np.concatenate([class_ids[0] * np.ones(int(len(XV)/2)), class_ids[1] * np.ones(int(len(XV)/2))])
                
                # print(XT.shape, XV.shape, yT.shape, yV.shape)
                
                st = time()
                
                if approach['option'] == 'classic':
                    
                    XTF = filt.apply_filter(XT)
                    XVF = filt.apply_filter(XV)
                    
                    if filtering['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
                        bmin = f_low * dft_size_band
                        bmax = f_high * dft_size_band
                        XTF = XTF[:, :, bmin:bmax]
                        XVF = XVF[:, :, bmin:bmax]
                        # print(bmin, bmax)
                    
                    # Option 1
                    csp.fit(XTF, yT)
                    XT_CSP = csp.transform(XTF)
                    XV_CSP = csp.transform(XVF) 
                    clf_final.fit(XT_CSP, yT)
                    scores = clf_final.predict(XV_CSP)
                    csp_filters = csp.filters_
                    
                    # Option 2
                    # chain = Pipeline([('CSP', csp), ('SVC', clf_final)])
                    # chain.fit(XTF, yT)
                    # scores = chain.predict(XVF)
                    # csp_filters = chain['CSP'].filters_
                    
                    if filtering['design'] == 'DFT': 
                        acc_dft = np.mean(scores == yV)
                        kappa_dft = cohen_kappa_score(scores, yV)
                    if filtering['design'] == 'IIR': 
                        acc_iir = np.mean(scores == yV)
                        kappa_iir = cohen_kappa_score(scores, yV)
                    
                    
                elif approach['option'] == 'sbcsp':
                    
                    nbands = int(approach['nbands'])
                    if nbands > (f_high-f_low): 
                        nbands = (f_high-f_low)
                        print(f'new nbands = {nbands}')
                    
                    n_bins = f_high - f_low
                    overlap = 0.5 if overlap else 1
                    step = n_bins / nbands
                    size = step / overlap
                    
                    sub_bands = []
                    for i in range(nbands):
                        fl_sb = round(i * step + f_low)
                        fh_sb = round(i * step + size + f_low)
                        if fh_sb <= f_high: sub_bands.append([fl_sb, fh_sb]) 
                        # se ultrapassar o limite superior da banda total, desconsidera a última sub-banda
                        # ... para casos em que a razão entre a banda total e n_bands não é exata 
                    
                    # print(sub_bands)
                    nbands = len(sub_bands)
                    
                    XTF, XVF = [], []
                    if filtering['design'] == 'DFT':
                        XT_FFT = filt.apply_filter(XT)
                        XV_FFT = filt.apply_filter(XV)
                        for i in range(nbands):
                            bmin = sub_bands[i][0] * dft_size_band
                            bmax = sub_bands[i][1] * dft_size_band
                            XTF.append(XT_FFT[:, :, bmin:bmax])
                            XVF.append(XV_FFT[:, :, bmin:bmax])
                            # print(bmin, bmax)
                            
                    elif filtering['design'] in ['IIR' or 'FIR']:
                        for i in range(nbands):
                            filt_sb = Filter(sub_bands[i][0], sub_bands[i][1], len(XT[0,0,:]), fs, filtering)
                            XTF.append(filt_sb.apply_filter(XT))
                            XVF.append(filt_sb.apply_filter(XV))
                    
                    chain = [ Pipeline([('CSP', CSP(n_components=ncomp)), ('LDA', LDA())]) for i in range(nbands) ]
                    
                    for i in range(nbands): chain[i]['CSP'].fit(XTF[i], yT)
                        
                    XT_CSP = [ chain[i]['CSP'].transform(XTF[i]) for i in range(nbands) ]
                    XV_CSP = [ chain[i]['CSP'].transform(XVF[i]) for i in range(nbands) ]
                    
                    SCORE_T = np.zeros((len(XT), nbands))
                    SCORE_V = np.zeros((len(XV), nbands))
                    for i in range(len(sub_bands)): 
                        chain[i]['LDA'].fit(XT_CSP[i], yT)
                        SCORE_T[:, i] = np.ravel(chain[i]['LDA'].transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
                        SCORE_V[:, i] = np.ravel(chain[i]['LDA'].transform(XV_CSP[i]))
                    
                    csp_filters_sblist = [chain[i]['CSP'].filters_ for i in range(nbands)]
                    lda_sblist = [chain[i]['LDA'] for i in range(nbands)] 
                    
                    SCORE_T0 = SCORE_T[yT == class_ids[0], :]
                    SCORE_T1 = SCORE_T[yT == class_ids[1], :]
                    p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
                    p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
                    META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
                    META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
            
                    clf_final.fit(META_SCORE_T, yT)
                    scores = clf_final.predict(META_SCORE_V)
                    
                    if filtering['design'] == 'DFT': 
                        acc_dft = np.mean(scores == yV)
                        kappa_dft = cohen_kappa_score(scores, yV)
                    if filtering['design'] == 'IIR': 
                        acc_iir = np.mean(scores == yV)
                        kappa_iir = cohen_kappa_score(scores, yV)
                
                if filtering['design'] == 'DFT': cost_dft = time() - st
                if filtering['design'] == 'IIR': cost_iir = time() - st
                
        if crossval: print(cross_scores)
        
        if cenario==0: RES_CLASSIC.append([suj, class_ids, tmin, tmax, round(acc_iir*100,2), round(kappa_iir,3), round(cost_iir, 2), round(acc_dft*100,2), round(kappa_dft,3), round(cost_dft, 2)])
        else: RES_SBCSP.append([suj, class_ids, tmin, tmax, round(acc_iir*100,2), round(kappa_iir,3), round(cost_iir, 2), round(acc_dft*100,2), round(kappa_dft,3), round(cost_dft, 2)])
        
RES_CLASSIC = pd.DataFrame(RES_CLASSIC, columns=['subj', 'classes', 'tmin', 'tmax', 'iir_acc', 'iir_kappa', 'iir_cost', 'dft_acc', 'dft_kappa', 'dft_cost'])
RES_SBCSP = pd.DataFrame(RES_SBCSP, columns=['subj', 'classes', 'tmin', 'tmax', 'iir_acc', 'iir_kappa', 'iir_cost', 'dft_acc', 'dft_kappa', 'dft_cost'])

S['classic_iir_acc'] = RES_CLASSIC['iir_acc']
S['classic_iir_kappa'] = RES_CLASSIC['iir_kappa']
S['classic_iir_cost'] = RES_CLASSIC['iir_cost']
S['classic_dft_acc'] = RES_CLASSIC['dft_acc']
S['classic_dft_kappa'] = RES_CLASSIC['dft_kappa']
S['classic_dft_cost'] = RES_CLASSIC['dft_cost']

S['sbcsp_iir_acc'] = RES_SBCSP['iir_acc']
S['sbcsp_iir_kappa'] = RES_SBCSP['iir_kappa']
S['sbcsp_iir_cost'] = RES_SBCSP['iir_cost']
S['sbcsp_dft_acc'] = RES_SBCSP['dft_acc']
S['sbcsp_dft_kappa'] = RES_SBCSP['dft_kappa']
S['sbcsp_dft_cost'] = RES_SBCSP['dft_cost']

pd.to_pickle(S, '/home/vboas/cloud/auto_setup_results/' + ds + '_s2_cortex/setup_list_final.pickle')
