# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
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
        # return self # instruction add because cross-validation pipeline
    def transform(self, X):        
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        XVAR = np.log(np.mean(XT ** 2, axis=2)) # Xcsp
        return XVAR

if __name__ == "__main__":
    # Classes = LH, RH, FooT, TonGue
    # Nchannels = 60
    # Nsessions = 2 (_T, _E)
    #    K3 -> 360 trials (90 per class)
    #    K6,L1 -> 240 trials (60 per class)
    # Subjects = 3 ('K3','K6','L1')
    # Fs = 250Hz
    # Timestamps Protocol: startTrial=0; beep/cross=2; startCue=3; startMI=4; endMI=7; endTrial=10 
    # Samplestamps Protocol: startTrial=0; Cue=750; startMI=1000; endMI=1750; endTrial=2500 
    subjects = ['K3','K6','L1'] 
    cls = [1, 2]
    W_Start = int(3 * 250)
    W_End = int(7 * 250)
    ACC = []
    for n_comp in [8, 16, 24, 32, 40, 48, 60]:
        #print("\n")
        for suj in subjects:
            XT = list(np.load('/mnt/dados/datasets/eeg_epochs/BCI3_3a/' + suj + '_T.npy')) # XT[0]=LH, XT[1]=RH, XT[2]=FT, XT[3]=TG
            XV = list(np.load('/mnt/dados/datasets/eeg_epochs/BCI3_3a/' + suj + '_E.npy')) # XV[0]=LH, XV[1]=RH, XV[2]=FT, XV[3]=TG
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
            clf = SVC(kernel="poly", C=10**(-4))
            clf.fit(XT_CSP, y)
            scores = clf.predict(XV_CSP)
            
            acc = np.mean(scores == y)

            #print(suj, n_comp, str(round(acc * 100, 1))+'%')
            ACC.append([suj,n_comp,round(acc * 100, 1)])
     
    FINAL = pd.DataFrame(ACC, columns=['Suj','M','Acc'])
    
    Mean = []
    Median = []
    Std = []
    # Correlação
    plt.figure(figsize=(10, 7), facecolor='mintcream')
    plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    plt.title("Desempenho do SVM na classificação de MI RH x LH para diferentes números de componentes")
    #plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
    plt.xlabel('N componentes', size=14)
    plt.ylabel('Acurácia (%)', size=14)
    plt.yscale('linear')
    plt.yticks(np.arange(40, 100, step=5))
    plt.xticks(np.unique(FINAL['M']))
    for suj in subjects:
        suj_data = FINAL.iloc[np.nonzero(FINAL['Suj']==suj)]
        plt.plot(suj_data['M'], suj_data['Acc']*100, lw=1)
        plt.scatter(suj_data['M'], suj_data['Acc']*100, label=(suj))
        Mean.append(suj_data['Acc'].mean())
        Median.append(suj_data['Acc'].median())
        Std.append(suj_data['Acc'].std())
    plt.legend(loc='best', ncol = 3, fontsize=11)
    plt.savefig('ml_bci33a_svm.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    