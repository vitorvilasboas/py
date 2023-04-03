# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import pickle
import itertools
import numpy as np
import pandas as pd
from time import time
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.linalg import eigh
from hyperopt import base, tpe, fmin, hp
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold

CFG = 'sbcsp' # 'csp_lda' or 'sbcsp'
FILTRO = 'FFT' # 'IIR' or 'FFT'
fl = 0 
fh = 50
opt_iter = 25

n_channels = 118
folds = 10
fs = 100
order = 5
t_start,t_end = 0.5,2.5
subjects = ['aa','al','av','aw','ay']
result = []

"""
    Classes = RH, FooT
    Nchannels = 118
    Nsessions = 1 -> 280 trials (140 per class)
        Epoch distribution:
    		aa : train=168 test=112  
        	al : train=224 test=56
        	av : train=84  test=196
        	aw : train=56  test=224
        	ay : train=28  test=252
    Subjects = 5 (aa, al, av, aw, ay)
    Fs = 100Hz
    Timestamps Protocol: startTrial=0; startCue=0; startMI=0; endTrial=5
    Samplestamps Protocol: startTrial=0; Cue/startMI=0; endTrial=500
"""

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
    n_comp = args
    
    nf = fs/2.
    bin0 = int(fl * (fs/nf))
    binN = int(fh * (fs/nf))
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    #test_size = 1/folds
    
    RES = []
    for suj in subjects:    
        X = list(np.load('/mnt/dados/eeg_data/BCI3_4a/npy_epochs/' + suj + '.npy')) 
                
        X = np.concatenate([X[0],X[1]]) # X[0]=RH, X[1]=FT
        y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) # vetor gabarito
                       
        X = X[:,:,W_Start:W_End]
        
        start = time() # start timer
        
        if FILTRO == 'FFT':
            # Spectral filtering
            XFF = fft(X)
            REAL = np.transpose(np.real(XFF)[:,:,bin0:binN], (2, 0, 1))
            IMAG = np.transpose(np.imag(XFF)[:,:,bin0:binN], (2, 0, 1)) 
            XF0 = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
            XF = np.transpose(XF0, (1, 2, 0))     
        elif FILTRO == 'IIR':
            # Temporal filtering
            b, a = butter(order, [fl/nf, fh/nf], btype='bandpass')
            XF = lfilter(b, a, X)
           
        ## Cross-validation:
        process = Pipeline([('CSP', CSP(n_comp)), ('classifier', LinearDiscriminantAnalysis())])
        #kf = StratifiedShuffleSplit(folds, test_size=test_size, random_state=0)
        kf = StratifiedKFold(folds, False)
        scores = cross_val_score(process, XF, y, cv=kf)
        
        cost = time() - start
        acc = np.mean(scores)
        
        RES.append([acc,cost])
        #print ('>>', suj, n_comp, str(round(acc*100,2))+'%', str(round(cost,1))+'s')
    
    #print(np.asarray(RES)[:,0].mean()*100, np.asarray(RES)[:,1].mean()*1000, np.asarray(RES)[:,1].sum())    
    return np.asarray(RES)
        
          
def sbcsp(args):
    print(args)
    n_comp, n_bands, Clog = args
    
    nf = fs/2.
    bin0 = int(fl * (fs/nf))
    binN = int(fh * (fs/nf))
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    #test_size = 1/folds
    
    subjects = ['aa','al','av','aw','ay'] 
    
    RES = []
    for suj in subjects:    
        X = list(np.load('/mnt/dados/eeg_data/BCI3_4a/npy_epochs/' + suj + '.npy')) 
                
        X = np.concatenate([X[0],X[1]]) # X[0]=RH, X[1]=FT
        y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) # vetor gabarito
                       
        X = X[:,:,W_Start:W_End]
        
        # XT = np.r_[X[:70,:,:],X[140:210,:,:]]
        # XV = np.r_[X[70:140,:,:],X[210:,:,:]]
        # y = np.concatenate([np.zeros(int(len(XT)/2)), np.ones(int(len(XT)/2))])
        
        #kf = StratifiedShuffleSplit(folds, test_size=test_size, random_state=0)
        kf = StratifiedKFold(folds, False)
        
        start = time() # start timer
        
        cross_scores = []
        for idx_treino, idx_teste in kf.split(X, y):
            XT = X[idx_treino]
            XV = X[idx_teste]
            yT = y[idx_treino]
            yV = y[idx_teste]
            
            if FILTRO == 'FFT':
                # Only Spectral filtering
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
            if FILTRO == 'FFT': n_bins = len(XT_FFT[0,0,:]) # ou (fh-fl) * 4 # Número total de bins de frequencia 
            elif FILTRO == 'IIR': n_bins = fh-fl 
            overlap = 2
            step = int(n_bins / n_bands)
            size = int(step * overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
    
            # Make sub-bands limits and Temporal/Spectral filtering
            SCORE_T = np.zeros((len(XT), n_bands))
            SCORE_V = np.zeros((len(XV), n_bands))
            for i in range(n_bands):
                if FILTRO == 'FFT':
                    # Only Spectral filtering
                    bin_ini = i*step
                    bin_fim = i*step+size
                    if bin_fim >= n_bins: bin_fim = n_bins-1
                    XTF = XT_FFT[:, :, bin_ini:bin_fim]
                    XVF = XV_FFT[:, :, bin_ini:bin_fim]
                elif FILTRO == 'IIR':
                    # Only Temporal filtering
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

        RES.append([np.mean(cross_scores),cost])
        #print ('>>', suj, n_comp, n_bands, str(round(np.mean(cross_scores)*100,2))+'%', str(round(cost,1))+'s')
    
    #print(np.asarray(RES)[:,0].mean()*100, np.asarray(RES)[:,1].mean()*1000, np.asarray(RES)[:,1].sum())    
    return np.asarray(RES)


def f_objective(args):
    global result
    if CFG == 'csp_lda':
        n_comp = args
        args2 = (int(n_comp))
        result = csp_lda(args2)
    elif CFG == 'sbcsp':
        n_comp, n_bands, Clog = args
        args2 = (int(n_comp), int(n_bands), Clog)    
        result = sbcsp(args2)
    acc_media = np.mean(result[:,0])
    #print(args,-acc_media)
    return -acc_media

if __name__ == "__main__":    
    if CFG == 'csp_lda':
        print(f_objective((13))) # para teste unitário
        space = (hp.quniform('n_comp', 2, 50, 2)) # 2 até n_channels salto=2
        #space = ((hp.choice('n_comp', range(2, 50, 2))))
    
    elif CFG == 'sbcsp': # ('n_bands', 1, (fh-fl)-1, 1) 
        #print(f_objective((4, 23, -4))) # para teste unitário
        space = (hp.quniform('n_comp', 2, 8, 2),
                  hp.quniform('n_bands', 28, 28, 1), 
                  hp.quniform('Clog', -4, -4, 1))
                  #hp.choice('Clog', [-4.])) #,hp.quniform('m', 20, 45, 1))
        
    # path_out = '/home/vboas/cloud/devto/bci-exp1/results/opt_bci3-4a_' + CFG + '_' + FILTRO + '_' + str(fl) + '-' + str(fh)
    # try:
    #     print('Trying to pickle file')
    #     #trials = base.Trials()
    #     trials = pickle.load(open(path_out, 'rb'))
    # except:
    #     print('No trial file at specified path, creating new one')
    #     trials = base.Trials()
    # else:
    #     print('File found')

    # try:
    #     print('Size of object: ' + str(len(trials)))
    #     best = fmin(f_objective, space=space, algo=tpe.suggest, max_evals=len(trials) + opt_iter, trials=trials)
    #     print(best)
    #     pickle.dump(trials, open(path_out, 'wb'))
    # except:
    #     print('Exception raised')
    #     #print(trials.)
    #     pickle.dump(trials, open(path_out, 'wb'))
    #     raise






    
#teste = pickle.load(open(path_out, 'rb'))

# for n_bands in sub_bands:
#     for n_comp in csp_ncomp:
#         args = (fs, fl, fh, order, t_start, t_end, n_channels, n_comp, n_bands, Clog, folds, filtro)
#         RES1 = f_objective(args)
#         RES_NCSP.append([n_comp,n_bands,np.asarray(np.asarray(RES)[:,3], dtype=float).mean(),np.asarray(np.asarray(RES)[:,4], dtype=float).mean()])     
#     RES_SB.append([n_bands,np.asarray(np.asarray(RES_NCSP)[:,2], dtype=float).mean(),np.asarray(np.asarray(RES_NCSP)[:,3], dtype=float).mean()])
#     RES_NCSP_FINAL = pd.DataFrame(np.asarray(RES_NCSP), columns=['Ncomps','Nbands','Acc','Time'])
#     RES_SB_FINAL = pd.DataFrame(np.asarray(RES_SB), columns=['Nbands','Acc','Time'])
# RES_FINAL = pd.DataFrame(RES1, columns=['Suj','Ncomps','Nbands','Acc','Time'])
    