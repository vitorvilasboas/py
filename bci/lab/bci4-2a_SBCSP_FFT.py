# -*- coding: utf-8 -*-
import os
import itertools
import pickle as pck
import numpy as np
import pandas as pd
from time import time
from scipy.fftpack import fft
from scipy.linalg import eigh
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
    csp_ncomp = [6]
    fs = 250
    fl = 8
    fh = 30
    nf = fs/2.
    # sub_bands = range(1,(fh-fl)+1)
    sub_bands = [10]
    bin0 = int(fl * (fs/nf)) # pos inicial dos bins de interesse 
    binN = int(fh * (fs/nf)) # pos final dos bins de interesse 
    Clog = -4
    t_start,t_end = 2.5,4.5
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    
    FBCSP_FFT = []
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
                           
                    # Temporal/Spectral filtering
                    XTF, XVF = [], []
                    for i in range(len(XTJ)):  
                        filtered = fft(XTJ[i])
                        REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1)) # transpose to intercalate
                        IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1)) 
                        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) # intercalate
                        filtered = np.transpose(filtered, (1, 2, 0)) # return to original format      
                        XTF.append(filtered)
                        
                        filtered = fft(XVJ[i])
                        REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
                        IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1))
                        filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
                        filtered = np.transpose(filtered, (1, 2, 0))       
                        XVF.append(filtered)
                    
                    # Concatenating training and evaluating data
                    XTF = np.concatenate([XTF[0],XTF[1]]) # Classes A and B - Training data [ ne*2 x nc x n_bins ]
                    XVF = np.concatenate([XVF[0],XVF[1]]) # Classes A and B - Evaluate data 
                    y = np.concatenate([np.zeros(int(len(XTF)/2)), np.ones(int(len(XTF)/2))]) # target vector
                    
                    # Divide sub-bands
                    n_bins = len(XTF[0,0,:]) # Número total de bins de frequencia
                    overlap = 2
                    step = int(n_bins / n_bands) # n bins por sub-bandas. i.e: se step = 8 então ex. [0.5 1 1.5 2 2.5 3 3.5 4]
                    size = int(step * overlap) # cada bin 2 entradas (real,imag) - tamanho fixo p/ todas sub bandas. overlap em 50%

                    XTSB, XVSB = [], []
                    for i in range(n_bands):
                        bin_ini = i*step
                        bin_fim = i*step+size
                        #print(bin_ini,'-',bin_fim,'-',n_bins-1,'-',XTF[0, 0, bin_ini:bin_fim])
                        if bin_fim >= n_bins: bin_fim = n_bins-1
                        sb = XTF[:, :, bin_ini:bin_fim]
                        XTSB.append(sb)
                        sb = XVF[:, :, bin_ini:bin_fim]
                        XVSB.append(sb)
                        nb.append(np.array([bin_ini/4,bin_fim/4])) # armazena freq_ini e freq_fim
                    
                    # CSP
                    csp = [CSP(n_components=n_comp) for i in range(n_bands)]
                    for i in range(n_bands): csp[i].fit(XTSB[i], y)
                    XT_CSP = [csp[i].transform(XTSB[i]) for i in range(n_bands)]
                    XV_CSP = [csp[i].transform(XVSB[i]) for i in range(n_bands)]
                    
                    # LDA
                    SCORE_TS = np.zeros((144, n_bands))
                    SCORE_VS = np.zeros((144, n_bands))
                    SCORE_T = np.zeros((144, n_bands))
                    SCORE_V = np.zeros((144, n_bands))
                    clf = [LinearDiscriminantAnalysis() for i in range(n_bands)]
                    svc1 = [SVC(kernel="rbf", C=10**Clog) for i in range(n_bands)]
                    for i in range(n_bands):
                        clf[i].fit(XT_CSP[i], y)
                        SCORE_T[:, i] = np.ravel(clf[i].transform(XT_CSP[i])) # classificaçoes de cada época nas N sub bandas - auto validação
                        SCORE_V[:, i] = np.ravel(clf[i].transform(XV_CSP[i])) # validação
                        
                        svc1[i].fit(XT_CSP[i], y)
                        SCORE_TS[:, i] = np.ravel(svc1[i].decision_function(XT_CSP[i])) # classificaçoes de cada época nas N sub bandas - auto validação
                        SCORE_VS[:, i] = np.ravel(svc1[i].decision_function(XV_CSP[i])) # validação
                    
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
                
                    FBCSP_FFT.append([suj,str(cls),n_comp,n_bands,acc,cost])
                
    FBCSP_FFT = pd.DataFrame(FBCSP_FFT, columns=['Subj','Classes','N CSP','N Sbands','Acc','Cost'])
    
    #pd.to_pickle(FBCSP_FFT, os.path.dirname(__file__) + '/FBCSP_FFT.pickle')
    #RESULTS = pd.read_pickle(os.path.dirname(__file__) + '/FBCSP_FFT.pickle')
    
    print('Média Total: {}%'.format(round(np.mean(FBCSP_FFT['Acc'])*100,1)))
    print('Custo Total: {}s'.format(round(np.sum(FBCSP_FFT['Cost']),2)))
    print('Custo Médio: {}ms'.format(round(np.mean(FBCSP_FFT['Cost']),2)))

    