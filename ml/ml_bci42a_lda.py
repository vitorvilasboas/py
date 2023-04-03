# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

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
        # return self # used on cross-validation pipeline
    def transform(self, X):        
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        XVAR = np.log(np.mean(XT ** 2, axis=2)) # Xcsp
        return XVAR
 
if __name__ == "__main__":
    # Classes = LH, RH, FooT, TonGue
    # Nchannels = 22
    # Nsubjects = 9 (A01,A02,A03, ... ,A09)
    # Nsessions = 2 (_T, _E) 
    #   2 * 288 trials = 576 total trials -> 2*72 = 144 per class -> 4*72 = 288 per session
    # Fs= 250Hz   
    # Timestamps Protocol: startTrial=0; cue=2; startMI=3.25; endMI=6; endTrial=7.5-8.5
    # Samplestamps Protocol: startTrial=0; cue=500; startMI=813; endMI=1500; endTrial=1875
    sujeitos = range(1,10)
    cls = [1, 2]
    W_Start = int(2.5 * 250)
    W_End = int(4.5 * 250)
    ACC = []
    for n_comp in [4,8,12,16,22]:
        for suj in sujeitos:
            XT = np.load('/mnt/dados/datasets/eeg_epochs/BCI4_2a/A0' + str(suj) + 'T.npy')
            XV = np.load('/mnt/dados/datasets/eeg_epochs/BCI4_2a/A0' + str(suj) + 'E.npy')
            XT = ([XT[cls[0]-1], XT[cls[1]-1]])
            XV = ([XV[cls[0]-1], XV[cls[1]-1]])
            
            XTJ = [ XT[i][:,:,W_Start:W_End] for i in range(len(XT)) ]
            XVJ = [ XV[i][:,:,W_Start:W_End] for i in range(len(XV)) ]
            
            b, a = butter(5, [8/125, 30/125], btype='bandpass')
            XTF = [ lfilter(b, a, XTJ[i]) for i in range(len(XTJ)) ]
            XVF = [ lfilter(b, a, XVJ[i]) for i in range(len(XVJ)) ]
            
            XTF = np.concatenate([XTF[0],XTF[1]]) # Classes A and B - Training data
            XVF = np.concatenate([XVF[0],XVF[1]]) # Classes A and B - Evaluate data
            y = np.concatenate([np.zeros(int(len(XTF)/2)), np.ones(int(len(XTF)/2))]) # target vector
            
            csp = CSP(n_components=n_comp)
            csp.fit(XTF, y)
            XT_CSP = csp.transform(XTF)
            XV_CSP = csp.transform(XVF)
            
            clf = LinearDiscriminantAnalysis()    
            #clf = SVC(kernel="poly", C=10**(-4))
            clf.fit(XT_CSP, y)
            scores = clf.predict(XV_CSP)
            
            acc = np.mean(scores == y)
            print(suj, n_comp, str(round(acc * 100, 2))+'%')
            
            ACC.append([suj, n_comp, round(acc * 100, 2)])
    
    FINAL = pd.DataFrame(ACC, columns=['Suj', 'M', 'Acc'])
    
    
    Mean = []
    # Correlação
    plt.figure(figsize=(10, 7), facecolor='mintcream')
    plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    #plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
    plt.xlabel('N componentes', size=14)
    plt.ylabel('Acurácia (%)', size=14)
    plt.yscale('linear')
    plt.yticks(np.arange(40, 100, step=5))
    plt.xticks(np.unique(FINAL['M']))
    for suj in sujeitos:
        suj_data = FINAL.iloc[np.nonzero(FINAL['Suj']==suj)]
        plt.plot(suj_data['M'], suj_data['Acc']*100, lw=1)
        plt.scatter(suj_data['M'], suj_data['Acc']*100, label=('S{}' .format(suj+1)))
        Mean.append(suj_data['Acc'].mean())
    plt.legend(loc='best', ncol = 3, fontsize=11)
    plt.savefig('ml_bci42a_lda.png', format='png', dpi=300, transparent=True, bbox_inches='tight')