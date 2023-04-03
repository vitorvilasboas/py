# -*- coding: utf-8 -*-
import itertools
import numpy as np
from time import time
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.linalg import eigh
from scipy.signal import lfilter, butter
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
        ind = np.empty(c, dtype=int)
        ind[0::2] = np.arange(c - 1, c // 2 - 1, -1)
        ind[1::2] = np.arange(0, c // 2)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        # return self # used on cross-validation pipeline
    def transform(self, X):
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        XVAR = np.log(np.mean(XT ** 2, axis=2))
        return XVAR
    
    
if __name__ == "__main__":
    
    subjects = range(1,10)
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    n_channels = 22
    csp_ncomp = np.arange(2, n_channels+1, 2)
    
    suj = subjects[0] # set 0 a 9
    cl = classes[0] # set 0 a 5
    n_comp = 8 # set 2 a n_channels
    
    fs = 250
    fl = 8 
    fh = 30
    nf = fs/2.
    sub_bands = np.arange(1,(fh-fl)+1)
    
    n_bands = 11 # set
    
    order = 5
    Clog = -4
    
    bin0 = int(fl * (fs/nf)) # para fl = 8 bin0 = 15 
    binN = int(fh * (fs/nf)) # para fl = 8 bin0 = 15 
    
    t_start,t_end = 2.5,4.5
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    
    XT = np.load('/mnt/dados/datasets/eeg_epochs/BCI4_2a/A0' + str(suj) + 'T.npy')
    XV = np.load('/mnt/dados/datasets/eeg_epochs/BCI4_2a/A0' + str(suj) + 'E.npy')
    
    XT = ([XT[cl[0]-1], XT[cl[1]-1]])
    XV = ([XV[cl[0]-1], XV[cl[1]-1]])
    
    XT = np.concatenate([XT[0],XT[1]]) # Dados de treinamento das classes A, B
    XV = np.concatenate([XV[0],XV[1]]) # Dados de validação das classes A, B
    y = np.concatenate([np.zeros(int(len(XT)/2)), np.ones(int(len(XT)/2))]) # Vetor gabarito
    
    XTJ = XT[:,:,W_Start:W_End] 
    XVJ = XV[:,:,W_Start:W_End]
    
    start = time() # start timer
        
    # Only Spectral filtering
    filtered = fft(XTJ)
    REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
    IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1)) 
    filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    XT_FFT = np.transpose(filtered, (1, 2, 0))       
    filtered = fft(XVJ)
    REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
    IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1))
    filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
    XV_FFT = np.transpose(filtered, (1, 2, 0))
    
    # Divide sub-bands
    n_bins = fh-fl # Número total de bins de frequencia 
    overlap = 2
    step = int(n_bins / n_bands)
    size = int(step * overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
    
    # Make sub-bands limits and Temporal/Spectral filtering
    SCORE_T = np.zeros((144, n_bands))
    SCORE_V = np.zeros((144, n_bands))
    for i in range(n_bands):
        # Only Spectral filtering
        bin_ini = i*step
        bin_fim = i*step+size
        if bin_fim >= n_bins: bin_fim = n_bins-1
        XTF = XT_FFT[:, :, bin_ini:bin_fim]
        XVF = XV_FFT[:, :, bin_ini:bin_fim]
        
        # Only Temporal filtering
        # fl_sb = i*step+fl
        # fh_sb = i*step+size+fl
        # if fl_sb == 0: fl_sb = 0.001
        # if fh_sb > fh: fh_sb = fh
        # b, a = butter(order, [fl_sb/nf, fh_sb/nf], btype='bandpass') # to filt IIR
        # XTF = lfilter(b, a, XTJ) 
        # XVF = lfilter(b, a, XVJ)
        
        csp = CSP(n_components=n_comp)
        csp.fit(XTF, y)
        XT_CSP = csp.transform(XTF) 
        XV_CSP = csp.transform(XVF)

        clf = LinearDiscriminantAnalysis()
        clf.fit(XT_CSP, y)
        SCORE_T[:, i] = np.ravel(clf.transform(XT_CSP)) # classificaçoes de cada época nas N sub bandas - auto validação
        SCORE_V[:, i] = np.ravel(clf.transform(XV_CSP)) # validação
    
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

    print (suj, cl, n_comp, n_bands, str(round(acc * 100, 2))+'%', str(int(cost*1000))+'ms')