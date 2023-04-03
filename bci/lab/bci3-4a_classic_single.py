# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import itertools
import numpy as np
from scipy.fftpack import fft
from scipy.linalg import eigh
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold

class CSP():
    def __init__(self, n_components):
        self.n_components = n_components
        # self.filters_ = None
    def fit(self, X, y):
        e, c, s = X.shape
        classes = np.unique(y)   
        Xa = X[classes[0] == y,:,:]
        Xb = X[classes[1] == y,:,:]
        S0 = np.zeros((c, c)) 
        S1 = np.zeros((c, c))
        for epoca in range(int(e/2)):
            S0 += np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T) #covA Xa[epoca]
            S1 += np.dot(Xb[epoca,:,:], Xb[epoca,:,:].T) #covB Xb[epoca]
        [D, W] = eigh(S0, S0 + S1)
        ind = np.empty(c, dtype=int)
        ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
        ind[1::2] = np.arange(0, c // 2)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        return self # instruction add because cross-validation pipeline
    def transform(self, X):        
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        XVAR = np.log(np.mean(XT ** 2, axis=2)) # Xcsp
        return XVAR

if __name__ == "__main__":
    
    subjects = ['aa','al','av','aw','ay'] 
    classes = [1, 3] # only RH, FT 
    n_channels = 118
    csp_ncomp = np.arange(2, n_channels+1, 2)
    
    suj = subjects[1] # 0 a 4
    n_comp = 18 # 2 a n_channels

    fs = 100
    fl = 8 
    fh = 30
    nf = fs/2.
    
    ordem = 5
    b, a = butter(ordem, [fl/nf, fh/nf], btype='bandpass')
    
    bin0 = int(fl * (fs/nf)) # para fl = 8 bin0 = 15 
    binN = int(fh * (fs/nf)) # para fl = 8 bin0 = 15 
    
    t_start,t_end = 0,4
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    
    X = list(np.load('/mnt/dados/datasets/eeg_epochs/BCI3_4a/' + suj + '.npy')) 
            
    X = np.concatenate([X[0],X[1]]) # X[0]=RH, X[1]=FT
    y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) # vetor gabarito
                   
    X = X[:,:,W_Start:W_End]
    
    # Spectral filtering
    XFF = fft(X)
    REAL = np.transpose(np.real(XFF)[:,:,bin0:binN], (2, 0, 1))
    IMAG = np.transpose(np.imag(XFF)[:,:,bin0:binN], (2, 0, 1)) 
    XF0 = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    XF = np.transpose(XF0, (1, 2, 0))     
    
    # Temporal filtering
    # XF = lfilter(b, a, X)
       
    ## Cross-validation:
    process = Pipeline([('CSP', CSP(n_comp)), ('classifier', LinearDiscriminantAnalysis())])
    #kf = StratifiedShuffleSplit(10, test_size=0.1, random_state=0)
    kf = StratifiedKFold(10, False, 0)
    scores = cross_val_score(process, XF, y, cv=kf)
    
   
    acc = np.mean(scores)
    print(suj, round(acc*100,1))
    
    
    