# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import lfilter, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

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
    # Classes = RH, FooT
    # Nchannels = 118
    # Nsessions = 1 -> 280 trials (140 per class)
    #    Epoch distribution:
    #		aa : train=168 test=112  
    #    	al : train=224 test=56
    #    	av : train=84  test=196
    #    	aw : train=56  test=224
    #    	ay : train=28  test=252
    # Subjects = 5 (aa, al, av, aw, ay)
    # Fs = 100Hz
    # Timestamps Protocol: startTrial=0; startCue=0; startMI=0; endTrial=5
    # Samplestamps Protocol: startTrial=0; Cue/startMI=0; endTrial=500
    subjects = ['aa','al','av','aw','ay'] 
    classes = [1, 3] # only RH, FT 
    W_Start = int(0 * 100)
    W_End = int(4 * 100)
    folds = 10
    sizeTest = 0.2
    ACC = []
    for n_comp in [10, 20, 40, 60, 80, 100, 118]:
        for suj in subjects:
            X = list(np.load('/mnt/dados/datasets/eeg_epochs/BCI3_4a/' + suj + '.npy')) # XT[0]=RH, XT[1]=FT
            X = np.concatenate([X[0],X[1]])
            X = X[:,:,W_Start:W_End] 
            b, a = butter(5, [8/125, 30/125], btype='bandpass')
            X = lfilter(b, a, X)
            y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) # Criando vetor gabarito
            clf = LinearDiscriminantAnalysis()
            clf = SVC(kernel="poly", C=10**(-4))
            ## Cross-validation: 
            process = Pipeline([('CSP', CSP(n_comp)), ('classifier', clf)]) # executa uma sequencia de processamento com um classificador no final
            cv = StratifiedShuffleSplit(folds, test_size=sizeTest, random_state=42)
            scores = cross_val_score(process, X, y, cv=cv)
            acc = np.mean(scores)
            print(suj, n_comp, str(round(acc * 100, 1))+'%')
            ACC.append([suj,n_comp,round(acc*100,1)])
    FINAL = pd.DataFrame(ACC, columns=['Suj','M','Mean Acc'])
    
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
    for suj in subjects:
        suj_data = FINAL.iloc[np.nonzero(FINAL['Suj']==suj)]
        plt.plot(suj_data['M'], suj_data['Mean Acc']*100, lw=1)
        plt.scatter(suj_data['M'], suj_data['Mean Acc']*100, label=(suj))
        Mean.append(suj_data['Mean Acc'].mean())
    plt.legend(loc='best', ncol = 3, fontsize=11)
    plt.savefig('ml_bci34a_lda.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    