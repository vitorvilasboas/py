# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import itertools
import numpy as np
from time import time
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.linalg import eigh
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold

subjects = ['aa','al','av','aw','ay']
n_channels = 118
fs = 100
nf = fs/2.

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


def csp_lda(args):
    filtro, order, fl, fh, t_start, t_end, n_comp, n_bands, Clog, folds, path = args
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    RES = []
    for suj in subjects:    
        X = list(np.load(path + suj + '.npy'))        
        X = np.concatenate([X[0],X[1]]) # X[0]=RH, X[1]=FT
        y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) # vetor gabarito              
        X = X[:,:,W_Start:W_End]
        start = time() # start timer
        if filtro == 'FFT': # Spectral filtering
            XFF = fft(X)
            bin0 = int(fl * (fs/nf))
            binN = int(fh * (fs/nf))
            REAL = np.transpose(np.real(XFF)[:,:,bin0:binN], (2, 0, 1))
            IMAG = np.transpose(np.imag(XFF)[:,:,bin0:binN], (2, 0, 1)) 
            XF0 = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
            XF = np.transpose(XF0, (1, 2, 0))     
        elif filtro == 'IIR': # Temporal filtering
            if fl==0: fl=0.001
            b, a = butter(order, [fl/nf, fh/nf], btype='bandpass')
            XF = lfilter(b, a, X)     
        process = Pipeline([('CSP', CSP(n_comp)), ('classifier', LinearDiscriminantAnalysis())])
        #kf = StratifiedShuffleSplit(folds, test_size=test_size, random_state=0)
        kf = StratifiedKFold(folds, False)
        scores = cross_val_score(process, XF, y, cv=kf) ## Cross-validation:
        cost = time() - start
        acc = np.mean(scores)
        RES.append([suj,acc,cost])
        print(suj, str(round(acc*100,2))+'%', str(round(cost,1))+'s')   
    return np.asarray(RES)
        
          
def sbcsp(args):
    filtro, order, fl, fh, t_start, t_end, n_comp, n_bands, Clog, folds, path = args
    bin0 = int(fl * (fs/nf))
    binN = int(fh * (fs/nf))
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    RES = []
    for suj in subjects:    
        X = list(np.load(path + suj + '.npy'))         
        X = np.concatenate([X[0],X[1]]) # X[0]=RH, X[1]=FT
        y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) # vetor gabarito              
        X = X[:,:,W_Start:W_End]
        kf = StratifiedKFold(folds, False)
        start = time() # start timer
        cross_scores = []
        for idx_treino, idx_teste in kf.split(X, y):
            XT = X[idx_treino]
            XV = X[idx_teste]
            yT = y[idx_treino]
            yV = y[idx_teste]
            if filtro == 'FFT': # Only Spectral filtering
                filtered = fft(XT)
                REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
                IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1)) 
                filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
                XT_FFT = np.transpose(filtered, (1, 2, 0))       
                filtered = fft(XV)
                REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
                IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1))
                filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
                XV_FFT = np.transpose(filtered, (1, 2, 0))
            # Divide sub-bands
            if filtro == 'FFT': n_bins = len(XT_FFT[0,0,:]) # ou (fh-fl) * 4 # Número total de bins de frequencia 
            elif filtro == 'IIR': n_bins = fh-fl 
            overlap = 2
            step = int(n_bins / n_bands)
            size = int(step * overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
            # Make sub-bands limits and Temporal/Spectral filtering
            SCORE_T = np.zeros((len(XT), n_bands))
            SCORE_V = np.zeros((len(XV), n_bands))
            for i in range(n_bands):
                if filtro == 'FFT': #Spectral filtering
                    bin_ini = i*step
                    bin_fim = i*step+size
                    if bin_fim >= n_bins: bin_fim = n_bins-1
                    XTF = XT_FFT[:, :, bin_ini:bin_fim]
                    XVF = XV_FFT[:, :, bin_ini:bin_fim]
                elif filtro == 'IIR': # Temporal filtering
                    fl_sb = i*step+fl
                    fh_sb = i*step+size+fl
                    if fl_sb == 0: fl_sb = 0.001
                    if fh_sb > fh: fh_sb = fh
                    #print(fl_sb, fh_sb, nf, fl_sb/nf, fh_sb/nf)
                    Wnl = fl_sb/nf
                    Wnh = fh_sb/nf
                    if Wnh >= 1: Wnh = 0.99
                    b, a = butter(order, [Wnl, Wnh], btype='bandpass') # to filt IIR
                    XTF = lfilter(b, a, XT) # comment here
                    XVF = lfilter(b, a, XV) # comment here
                csp = CSP(n_components=n_comp)
                csp.fit(XTF, yT)
                XT_CSP = csp.transform(XTF) 
                XV_CSP = csp.transform(XVF)
                clf = LinearDiscriminantAnalysis()
                clf.fit(XT_CSP, yT)
                SCORE_T[:, i] = np.ravel(clf.transform(XT_CSP)) # classificaçoes de cada época nas N sub bandas - auto validação
                SCORE_V[:, i] = np.ravel(clf.transform(XV_CSP)) # validação
            # Meta-classificador Bayesiano
            SCORE_T0 = SCORE_T[yT == 0, :]
            m0 = np.mean(SCORE_T0, axis=0) #media classe A
            std0 = np.std(SCORE_T0, axis=0) #desvio padrão classe A
            SCORE_T1 = SCORE_T[yT == 1, :]
            m1 = np.mean(SCORE_T1, axis=0)
            std1 = np.std(SCORE_T1, axis=0)
            p0 = norm(m0, std0) # p0 e p1 representam uma distribuição normal de médias m0 e m1, e desvio padrão std0 e std1
            p1 = norm(m1, std1)
            META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
            META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
            # SVM on top of the meta-classifier
            svc = SVC(kernel="linear", C=10**Clog)
            svc.fit(META_SCORE_T, yT)
            scores = svc.predict(META_SCORE_V)
            cost = time() - start
            acc = np.mean(scores == yV)
            cross_scores.append(acc)

        RES.append([suj,np.mean(cross_scores),cost])
        print(suj, str(round(np.mean(cross_scores)*100,2))+'%', str(round(cost,1))+'s')    
    return np.asarray(RES)


if __name__ == "__main__":    
    path = '/mnt/dados/eeg_data/eeg_epochs/BCI3_4a/'
    filtro = 'FFT' #'IIR' or 'FFT'
    order = 5
    fl = 5
    fh = 35
    n_comp = 6
    n_bands = 8
    Clog = -3
    t_start,t_end = 0.5,2.5
    folds = 10
    args = (filtro, order, fl, fh, t_start, t_end, n_comp, n_bands, Clog, folds, path)
    RES = csp_lda(args) #csp_lda or sbcsp 
    print(f'{str(round(np.asarray(RES[:,1], dtype=float).mean()*100,2))}%  +- {str(round(np.asarray(RES[:,1], dtype=float).std()*100,1))}% || {round(np.asarray(RES[:,2], dtype=float).sum(),1)}s')
    