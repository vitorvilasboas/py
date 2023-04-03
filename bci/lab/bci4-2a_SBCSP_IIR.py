# -*- coding: utf-8 -*-
import os
import pickle as pck
import numpy as np
import pandas as pd
from time import time
from scipy.linalg import eigh
from scipy.signal import lfilter, butter
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class CSP():
    def __init__(self, n_components):
        self.n_components = n_components
        # self.filters_ = None
    def fit(self, X, y):
        e, c, t = X.shape
        classes = np.unique(y)
        X0 = X[classes[0] == y,:,:]
        X1 = X[classes[1] == y,:,:]
        S0 = np.zeros((c, c)) # Sum up covariance matrix
        S1 = np.zeros((c, c))
        for i in range(int(e/2)):
            S0 += np.dot(X0[i,:,:], X0[i,:,:].T) #covA X0[epoca]
            S1 += np.dot(X1[i,:,:], X1[i,:,:].T) #covB X1[epoca]
        [D, W] = eigh(S0, S0 + S1)
        ind = np.empty(c, dtype=int) # usado para a ideia de pares de componentes (extremos juntos)
        ind[0::2] = np.arange(c - 1, c // 2 - 1, -1)
        ind[1::2] = np.arange(0, c // 2) #ind [21 0 20 1 19 2 ... 12 9 11 10]
        W = W[:, ind] # reorganizando W conforme ind 
        self.filters_ = W.T[:self.n_components] # considera somente as n_components colunas de W1 que passam a ser linhas com a transposição
        # return self # used on cross-validation pipeline
    def transform(self, X):
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        XVAR = np.log(np.mean(XT ** 2, axis=2))
        return XVAR
    
if __name__ == "__main__":
    subjects = range(1,10)
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    n_channels = 22
    # csp_ncomp = range(2, n_channels+1, 2)
    csp_ncomp = [8] #10
    fs = 250
    fl = 8
    fh = 30
    order = 5
    nf = fs/2.
    sub_bands = np.arange(1,(fh-fl)+1)
    #sub_bands = [11]
    Clog = -4
    t_start,t_end = 2.5,4.5
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    
    FBCSP_IIR = []
    for n_comp in csp_ncomp:
        for n_bands in sub_bands:
            nb = [] # apenas para aferição
            for cls in classes:
                #print("\n")
                for suj in subjects:
                    # Loading dataset
                    #XT = [ np.load(open('/mnt/dados/datasets/BCI4_2a/npy/A0' + str(suj) + 'T_' + str(cls[j]) + '.npy', 'rb')) for j in range(2) ]         
                    #XV = [ np.load(open('/mnt/dados/datasets/BCI4_2a/npy/A0' + str(suj) + 'E_' + str(cls[j]) + '.npy', 'rb')) for j in range(2) ]
                    
                    XT = list(np.load('/mnt/dados/datasets/eeg_epochs/BCI4_2a/A0' + str(suj) + 'T.npy'))
                    XV = list(np.load('/mnt/dados/datasets/eeg_epochs/BCI4_2a/A0' + str(suj) + 'E.npy'))
                    
                    XT = ([XT[cls[0]-1], XT[cls[1]-1]])
                    XV = ([XV[cls[0]-1], XV[cls[1]-1]])
    
                    # Windowing
                    XTJ = [ XT[i][:,:,W_Start:W_End] for i in range(len(XT)) ]
                    XVJ = [ XV[i][:,:,W_Start:W_End] for i in range(len(XV)) ]
    
                    start = time() # start timer
                
                    # Concatenating training and evaluating data
                    XTJ = np.concatenate([XTJ[0],XTJ[1]]) # Classes A and B - Training data [ ne*2 x nc x n_bins ]
                    XVJ = np.concatenate([XVJ[0],XVJ[1]]) # Classes A and B - Evaluate data 
                    y = np.concatenate([np.zeros(int(len(XTJ)/2)), np.ones(int(len(XTJ)/2))]) # target vector
                    
                    # Divide sub-bands
                    n_bins = fh-fl # Número total de bins de frequencia 
                    overlap = 2
                    step = int(n_bins / n_bands)
                    size = int(step * overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
                    
                    # Make sub-bands limits and Temporal/Spectral filtering
                    XTF = []
                    XVF = [] 
                    for i in range(n_bands):
                        fl_sb = i*step+fl
                        fh_sb = i*step+size+fl
                        if fl_sb == 0: fl_sb = 0.001
                        if fh_sb > fh: fh_sb = fh
                        nb.append(np.array([fl_sb,fh_sb])) # apenas para aferição
                
                        b, a = butter(order, [fl_sb/nf, fh_sb/nf], btype='bandpass') # to filt IIR
                        XTF.append( lfilter(b, a, XTJ) ) # Temporal/Spectral filtering
                        XVF.append( lfilter(b, a, XVJ) ) # o filtro é aplicado por padrão na última dimensão
                    
                    # CSP
                    csp = [CSP(n_components=n_comp) for i in range(n_bands)]
                    for i in range(n_bands): csp[i].fit(XTF[i], y)
                    XT_CSP = [csp[i].transform(XTF[i]) for i in range(n_bands)]
                    XV_CSP = [csp[i].transform(XVF[i]) for i in range(n_bands)]
                
                    # LDA
                    SCORE_T = np.zeros((144, n_bands))
                    SCORE_V = np.zeros((144, n_bands))
                    clf = [LinearDiscriminantAnalysis() for i in range(n_bands)]
                    for i in range(n_bands):
                        clf[i].fit(XT_CSP[i], y)
                        SCORE_T[:, i] = np.ravel(clf[i].transform(XT_CSP[i])) # classificaçoes de cada época nas N sub bandas - auto validação
                        SCORE_V[:, i] = np.ravel(clf[i].transform(XV_CSP[i])) # validação
                    
                    # Meta-classificador Bayesiano
                    SCORE_T0 = SCORE_T[y == 0, :]
                    m0 = np.mean(SCORE_T0, axis=0) #media classe A
                    std0 = np.std(SCORE_T0, axis=0) #desvio padrão classe A
                    
                    SCORE_T1 = SCORE_T[y == 1, :]
                    m1 = np.mean(SCORE_T1, axis=0)
                    std1 = np.std(SCORE_T1, axis=0)
                
                    p0 = norm(m0, std0) # p0 e p1 representam uma distribuição normal de médias m0 e m1, e desvio padrão std0 e std1
                    p1 = norm(m1, std1)
                    
                    META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
                    META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
                    
                    # SVM on top of the meta-classifier
                    svc = SVC(kernel="linear", C=10**Clog)
                    svc.fit(META_SCORE_T, y)
                    scores = svc.predict(META_SCORE_V)
                    
                    cost = time() - start # stop timer (cost estimate)
                    acc = np.mean(scores == y) 
                
                    # print (suj, cls, n_comp, n_bands, str(round(acc * 100, 2))+'%', str(int(cost*1000))+'ms')
                
                    FBCSP_IIR.append([suj,str(cls),n_comp,n_bands,acc,cost])
                
    FBCSP_IIR = pd.DataFrame(FBCSP_IIR, columns=['Subj','Classes','N CSP','N Sbands','Acc','Cost'])
    np.save(os.path.dirname(__file__) + '/results/' + 'FBCSP_IIR_8a30_r01.npy', FBCSP_IIR)
    pd.to_pickle(FBCSP_IIR, os.path.dirname(__file__) + '/results/' + '/FBCSP_IIR_8a30_r01.pickle')
    
    sb_means = np.asarray([ FBCSP_IIR.iloc[ np.where(FBCSP_IIR['N Sbands']==j) ]['Acc'].mean() for j in sub_bands ])
            
    SB_MEANS = pd.DataFrame(
         np.concatenate((np.ones((22,1))*csp_ncomp, sub_bands.reshape(-1,1), (sb_means*100).reshape(-1,1)), axis=1), 
         columns=['N CSP','N Sbands','Acc Média'])
    
    
    #    Best_Sb_NComp2 = pd.DataFrame(
#            np.concatenate([ncomp.reshape(11,1),
#                            Step2_nsb_best2.reshape(11,1) + 1,
#                            Step3_acc_best2.reshape(11,1)
#                            ], axis=1), columns=['N CSP','Best N Sb','Mean Acc'])
    
    ## Para exibiçãoa de resultados de uma única configuração (todos os hiperparâmetros definidos)
    print('Média Total: {}%'.format(round(np.mean(FBCSP_IIR['Acc'])*100,1)))
    print('Custo Total: {}s'.format(round(np.sum(FBCSP_IIR['Cost']),2)))
    print('Custo Médio: {}ms'.format(round(np.mean(FBCSP_IIR['Cost']),2)))
    
    
    ################ ANÁLISE ##########################
#    suj = np.ravel(range(1,10))
#    ncomp = np.ravel(range(2, 23, 2))
#    sbands = np.ravel(range(1, 23))
#    
#    R = pd.read_pickle(os.path.dirname(__file__) + '/results/' + '/FBCSP_IIR_8a30_r01.pickle')
#    
#    LH_RH = R.iloc[np.where(R['Classes']=='[1, 2]')]
#    
#    #N_CSP = [ [ R.iloc[ np.where((R['N CSP']==i)&(R['N Sbands']==j)) ] for j in sbands ] for i in ncomp ]
#    
#    # Melhores N sub-bandas para cada um dos pares de filtros CSP (Para 4 classes um_contra_um = 6 combinações)
#    Step1_mean = [ [ R.iloc[ np.where((R['N CSP']==i)&(R['N Sbands']==j)) ]['Acc'].mean() for j in sbands ] for i in ncomp ] 
#    Step2_nsb_best = np.ravel([ np.where(Step1_mean[i] == max(Step1_mean[i])) for i in range(len(Step1_mean)) ])
#    Step3_acc_best = np.asarray([ Step1_mean[i][j] for i,j in zip(range(len(Step1_mean)), Step2_nsb_best) ])
#    Best_Sb_NComp = pd.DataFrame(
#            np.concatenate([ncomp.reshape(11,1),
#                            Step2_nsb_best.reshape(11,1) + 1,
#                            Step3_acc_best.reshape(11,1)
#                            ], axis=1), columns=['N CSP','Best N Sb','Mean Acc'])
#    
#    # Melhores N sub-bandas para cada um dos pares de filtros CSP (Para 2 classes LHxRH)
#    Step1_mean2 = [ [ LH_RH.iloc[ np.where((LH_RH['N CSP']==i)&(LH_RH['N Sbands']==j)) ]['Acc'].mean() for j in sbands ] for i in ncomp ] 
#    Step2_nsb_best2 = np.ravel([ np.where(Step1_mean2[i] == max(Step1_mean2[i])) for i in range(len(Step1_mean2)) ])
#    Step3_acc_best2 = np.asarray([ Step1_mean2[i][j] for i,j in zip(range(len(Step1_mean2)), Step2_nsb_best2) ])
#    Best_Sb_NComp2 = pd.DataFrame(
#            np.concatenate([ncomp.reshape(11,1),
#                            Step2_nsb_best2.reshape(11,1) + 1,
#                            Step3_acc_best2.reshape(11,1)
#                            ], axis=1), columns=['N CSP','Best N Sb','Mean Acc'])
#     
#    # Média Acc por sujeito (4 classes)
#    X1 = R.iloc[ np.where((R['N CSP']==10)&(R['N Sbands']==10)) ]
#    MeanSuj1 = np.asarray([ X1.iloc[np.where(X1['Subj']==i)]['Acc'].mean() for i in suj ])
#    print('Média Total: ',np.mean(MeanSuj1))
#    
#    # Média Acc por sujeito (LH x RH)
#    X2 = LH_RH.iloc[ np.where((LH_RH['N CSP']==10)&(LH_RH['N Sbands']==10)) ]
#    MeanSuj2 = np.asarray([ X2.iloc[np.where(X2['Subj']==i)]['Acc'].mean() for i in suj ])
#    print('Média Total: ',np.mean(MeanSuj2)) # np.mean(X2['Acc'])
#    
#    S1 = R.iloc[np.nonzero(R['Subj']==1) # R.iloc[np.where(R['Subj']==1)]
#    #SumCost = R.iloc[np.where(R['Subj']==1) ]['Cost'].sum()
#    #MeanSB = np.asarray([ LH_RH.iloc[np.where(LH_RH['N Sbands']== i)].iloc[:,3:5].mean() for i in sbands])
