# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas

import math
import pickle
import numpy as np
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

ds = 'III4a' # III3a, III4a, IV2a, IV2b, LEE54, CL, TWL
fs = 1000.0 if ds=='LEE54' else 100 if ds=='III4a' else 125 if ds=='CL' else 250
suj = 'aa'
class_ids = [1, 3]
path = '/mnt/dados/eeg_data/' + ds + '/omi/'  # + 'A0' # 'subj'

crossval = False
overlap = True
nfolds = 10
test_perc = 0.1 if crossval else 0.5

f_low, f_high, ncomp, tmin, tmax = 8, 30, 4, 0.5, 2.5    
            
# clf = {'model':'Bayes'}
# clf = {'model':'LDA', 'lda_solver':'svd'} # 'lda_solver': 'svd','lsqr','eigen'
# clf = {'model':'KNN', 'metric':'chebyshev', 'neig':27} # 'metric': 'euclidean','manhattan','minkowski','chebyshev'
clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4} # 'kernel': 'linear', 'poly', 'sigmoid', 'rbf'
# clf = {'model':'MLP', 'eta':-3, 'activ':{'af':'tanh'}, 'alpha':-1, 'n_neurons':194, 'n_hidden':3, 'mlp_solver':'adam'} # 'mlp_solver':'adam', 'lbfgs', 'sgd' # 'af':'identity', 'logistic', 'tanh', 'relu'
# clf = {'model':'DTree', 'crit':'gini'} # 'crit': 'entropy' or 'gini'

# approach = {'option':'classic'}
approach = {'option':'sbcsp', 'nbands':15}

filtering = {'design':'DFT'}
# filtering = {'design':'IIR', 'iir_order':5}
# filtering = {'design':'FIR', 'fir_order':5}


###################################################

data, events, info = pickle.load(open(path + str(suj) + '.omi', 'rb'))

smin = math.floor(tmin * fs)
smax = math.floor(tmax * fs)
buffer_len = smax - smin

epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
epochs = nanCleaner(epochs)
# self.epochs = np.asarray([ nanCleaner(ep) for ep in self.epochs ])

filt = Filter(f_low, f_high, buffer_len, fs, filtering)

csp = CSP(n_components=ncomp)

cross_scores = None
cross_kappa = None

if clf['model'] == 'LDA': clf_final = LDA(solver=clf['lda_solver'], shrinkage=None)
if clf['model'] == 'Bayes': clf_final = GaussianNB()
if clf['model'] == 'SVM': clf_final = SVC(kernel=clf['kernel']['kf'], C=10 ** (clf['C']), gamma='scale', degree=3, probability=True)
if clf['model'] == 'KNN': clf_final = KNeighborsClassifier(n_neighbors=int(clf['neig']), metric=clf['metric'], p=3)           
if clf['model'] == 'DTree': clf_final = DecisionTreeClassifier(criterion=clf['crit'], random_state=0, max_depth=None, min_samples_split=2)  
if clf['model'] == 'MLP': clf_final = MLPClassifier(verbose=False, max_iter=1000, tol=0.0001, learning_rate_init=10**clf['eta'], alpha=10**clf['alpha'], 
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
        
        # chain = Pipeline([('CSP', csp), ('SVC', clf_final)])
        # cross_scores = cross_val_score(chain, XF, labels, cv=kf)
        
        for idx_treino, idx_teste in kf.split(epochs, labels):
            XT, XV, yT, yV = XF[idx_treino], XF[idx_teste], labels[idx_treino], labels[idx_teste]
            chain = Pipeline([('CSP', csp), ('SVC', clf_final)])
            chain.fit(XT, yT)
            csp_filters = chain['CSP'].filters_
            
            scores = chain.predict(XV)
            acc_fold = np.mean(scores == yV) # or cross_scores.append(chain.score(XV, yV))       
            cross_scores.append(acc_fold)
            cross_kappa.append( cohen_kappa_score(scores, yV) )
    
    
    
    elif approach['option'] == 'sbcsp':
        
        for idx_treino, idx_teste in kf.split(epochs, labels):
            
            XT, XV, yT, yV = epochs[idx_treino], epochs[idx_teste], labels[idx_treino], labels[idx_teste]
            
            chain = Pipeline([('CSP', CSP(n_components=ncomp)), ('LDA', LDA()), ('SVC', clf_final)])
            nbands = int(approach['nbands'])
            # nbands = (f_high-f_low)-1 if nbands >= (f_high-f_low) else nbands
            nbands = (f_high-f_low) if nbands > (f_high-f_low) else nbands
            
            if filtering['design'] == 'DFT':
                XT_FFT = filt.apply_filter(XT)
                XV_FFT = filt.apply_filter(XV)
                n_bins = len(XT_FFT[0, 0, :])  # ou (fh-fl) * 4 # Número total de bins de frequencia
            else: n_bins = f_high - f_low
            overlap = 0.5 if overlap else 1
            step = n_bins / nbands
            size = step / overlap 
            SCORE_T = np.zeros((len(XT), nbands))
            SCORE_V = np.zeros((len(XV), nbands))
            csp_filters_sblist = []
            lda_sblist = []
            for i in range(nbands):
                if filtering['design'] == 'DFT':

                    bin_ini = round(i * step)
                    bin_fim = round(i * step + size)
                    if bin_fim >= n_bins: bin_fim = n_bins # - 1
                    XTF = XT_FFT[:, :, bin_ini:bin_fim]
                    XVF = XV_FFT[:, :, bin_ini:bin_fim]
                    print( round(bin_ini * (filt.res_freq/2) + f_low), round(bin_fim * (filt.res_freq/2) + f_low) ) # print bins convertidos para Hertz
                    
                else:
                                       
                    fl_sb = round(i * step + f_low)
                    fh_sb = round(i * step + size + f_low)
                    if fh_sb > f_high: fh_sb = f_high
                    if fl_sb > fh_sb: fl_sb = fh_sb               
                    filt_sb = Filter(fl_sb, fh_sb, len(XT[0,0,:]), fs, filtering)
                    XTF = filt_sb.apply_filter(XT)
                    XVF = filt_sb.apply_filter(XV)
                    print(fl_sb, fh_sb)
                    
                chain['CSP'].fit(XTF, yT)
                XT_CSP = chain['CSP'].transform(XTF)
                XV_CSP = chain['CSP'].transform(XVF)
                chain['LDA'].fit(XT_CSP, yT)
                SCORE_T[:, i] = np.ravel(chain['LDA'].transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
                SCORE_V[:, i] = np.ravel(chain['LDA'].transform(XV_CSP))
                csp_filters_sblist.append(chain['CSP'].filters_)
                lda_sblist.append(chain['LDA'])
                
                # csp = CSP(n_components=ncomp)
                # lda = LDA()
                # csp.fit(XTF, yT)
                # XT_CSP = csp.transform(XTF)
                # XV_CSP = csp.transform(XVF)
                
                # lda.fit(XT_CSP, yT)
                # SCORE_T[:, i] = np.ravel(lda.transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
                # SCORE_V[:, i] = np.ravel(lda.transform(XV_CSP))
        
            SCORE_T0 = SCORE_T[yT == class_ids[0], :]
            SCORE_T1 = SCORE_T[yT == class_ids[1], :]
            p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
            p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
            META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
            META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
        
            chain['SVC'].fit(META_SCORE_T, yT)
            scores = chain['SVC'].predict(META_SCORE_V)
            
            # clf_final.fit(META_SCORE_T, yT)
            # scores = clf_final.predict(META_SCORE_V)
            
                                    
            acc_fold = np.mean(scores == yV) # calculo acurácia
                   
            cross_scores.append(acc_fold)
            
            cross_kappa.append( cohen_kappa_score(scores, yV) )
            
    acc = np.mean(cross_scores)  # acurácia média (todas nfolds iterações)
    kappa = np.mean(cross_kappa) 
    
    cost = time() - st
    
else: # not crossval
    
    test_perc = 0.5
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
        
        chain = Pipeline([('CSP', csp), ('SVC', clf_final)])
        XTF = filt.apply_filter(XT)
        XVF = filt.apply_filter(XV)
        
        # csp.fit(XTF, yT)
        # XT_CSP = csp.transform(XTF)
        # XV_CSP = csp.transform(XVF) 
        # svc_final.fit(XT_CSP, yT)
        # scores = svc_final.predict(XV_CSP)
        # acc = np.mean(scores == yV)
        
        chain.fit(XTF, yT)
        csp_filters = chain['CSP'].filters_
        
        scores = chain.predict(XVF)
        acc = np.mean(scores == yV) # or chain.score(XVF, yV)     
        kappa = cohen_kappa_score(scores, yV)
        
        
    
    elif approach['option'] == 'sbcsp':
        
        chain = Pipeline([('CSP', CSP(n_components=ncomp)), ('LDA', LDA()), ('SVC', clf_final)])
        nbands = int(approach['nbands'])
        # nbands = (f_high-f_low)-1 if nbands >= (f_high-f_low) else nbands
        nbands = (f_high-f_low) if nbands > (f_high-f_low) else nbands
        
        if filtering['design'] == 'DFT':
            XT_FFT = filt.apply_filter(XT)
            XV_FFT = filt.apply_filter(XV)
            n_bins = len(XT_FFT[0, 0, :])  # ou (fh-fl) * 4 # Número total de bins de frequencia
        else: n_bins = f_high - f_low
        
        
        overlap = 0.5 if overlap else 1
        step = n_bins / nbands
        size = step / overlap 
        SCORE_T = np.zeros((len(XT), nbands))
        SCORE_V = np.zeros((len(XV), nbands))
        csp_filters_sblist = []
        lda_sblist = []
        for i in range(nbands):
            
            if filtering['design'] == 'DFT':
                bin_ini = round(i * step)
                bin_fim = round(i * step + size)
                if bin_fim >= n_bins: bin_fim = n_bins # -1
                XTF = XT_FFT[:, :, bin_ini:bin_fim]
                XVF = XV_FFT[:, :, bin_ini:bin_fim]
                print( round(bin_ini * (filt.res_freq/2) + f_low), round(bin_fim * (filt.res_freq/2) + f_low) ) # print bins convertidos para Hertz
                
            else:
                fl_sb = round(i * step + f_low)
                fh_sb = round(i * step + size + f_low)
                if fh_sb > f_high: fh_sb = f_high
                if fl_sb > fh_sb: fl_sb = fh_sb
                filt_sb = Filter(fl_sb, fh_sb, len(XT[0,0,:]), fs, filtering)
                XTF = filt_sb.apply_filter(XT)
                XVF = filt_sb.apply_filter(XV) 
                print(fl_sb, fh_sb),
                
                
            chain['CSP'].fit(XTF, yT)
            XT_CSP = chain['CSP'].transform(XTF)
            XV_CSP = chain['CSP'].transform(XVF)
            chain['LDA'].fit(XT_CSP, yT)
            SCORE_T[:, i] = np.ravel(chain['LDA'].transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
            SCORE_V[:, i] = np.ravel(chain['LDA'].transform(XV_CSP))
            csp_filters_sblist.append(chain['CSP'].filters_)
            lda_sblist.append(chain['LDA'])
            
            # csp = CSP(n_components=ncomp)
            # lda = LDA()
            # csp.fit(XTF, yT)
            # XT_CSP = csp.transform(XTF)
            # XV_CSP = csp.transform(XVF)
            
            # lda.fit(XT_CSP, yT)
            # SCORE_T[:, i] = np.ravel(lda.transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
            # SCORE_V[:, i] = np.ravel(lda.transform(XV_CSP))
    
        SCORE_T0 = SCORE_T[yT == class_ids[0], :]
        SCORE_T1 = SCORE_T[yT == class_ids[1], :]
        p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
        p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
        META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
        META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
    
        chain['SVC'].fit(META_SCORE_T, yT)
        scores = chain['SVC'].predict(META_SCORE_V)
        
        # clf_final.fit(META_SCORE_T, yT)
        # scores = clf_final.predict(META_SCORE_V)
        
        acc = np.mean(scores == yV)
        kappa = cohen_kappa_score(scores, yV)

    cost = time() - st

print(suj, class_ids, str(round(acc*100,2))+'%', str(round(kappa,3)), str(round(cost, 2))+'s')