# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import itertools
import numpy as np
from time import time
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.linalg import eigh
from scipy.signal import lfilter, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

subjects = range(1,10)
n_channels = 22
classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
fs = 250
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
    filtro, order, fl, fh, t_start, t_end, n_comp, n_bands, Clog, path = args
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    bin0 = int(fl * (fs/nf))
    binN = int(fh * (fs/nf))
    RES = []
    for cl in classes:
        for suj in subjects:   
            XT = np.load(path + str(suj) + 'T.npy')
            XV = np.load(path + str(suj) + 'E.npy')
            XT = ([XT[cl[0]-1], XT[cl[1]-1]])
            XV = ([XV[cl[0]-1], XV[cl[1]-1]])
            XT = np.concatenate([XT[0],XT[1]]) # Dados de treinamento das classes A, B
            XV = np.concatenate([XV[0],XV[1]]) # Dados de validação das classes A, B
            y = np.concatenate([np.zeros(int(len(XT)/2)), np.ones(int(len(XT)/2))]) # Vetor gabarito
            XTJ = XT[:,:,W_Start:W_End] 
            XVJ = XV[:,:,W_Start:W_End]
            start = time() # start timer
            if filtro == 'FFT': # Spectral filtering
                filtered = fft(XTJ)
                REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
                IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1)) 
                filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
                XTF = np.transpose(filtered, (1, 2, 0))       
                filtered = fft(XVJ)
                REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
                IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1))
                filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
                XVF = np.transpose(filtered, (1, 2, 0))
            elif filtro == 'IIR': # Temporal filtering
                if fl==0: fl=0.001
                b, a = butter(order, [fl/nf, fh/nf], btype='bandpass') # to filt IIR
                XTF = lfilter(b, a, XTJ)
                XVF = lfilter(b, a, XVJ)
            csp = CSP(n_components=n_comp)
            csp.fit(XTF, y)
            XT_CSP = csp.transform(XTF)
            XV_CSP = csp.transform(XVF) 
            clf = LinearDiscriminantAnalysis()
            #clf = SVC(kernel="poly", C=10**(-4))
            #clf = KNeighborsClassifier(n_neighbors=24, metric='minkowski', p=2) #minkowski e p=2 -> para usar distancia euclidiana padrão
            #clf = DecisionTreeClassifier(criterion='entropy', random_state=0) #max_depth = None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
            #clf = GaussianNB()
            #clf = MLPClassifier(verbose=False, max_iter=10000, tol=0.0001, activation='logistic', learning_rate_init=0.001, learning_rate='invscaling',  solver='adam') #hidden_layer_sizes=(100,),
            clf.fit(XT_CSP, y)
            scores = clf.predict(XV_CSP)
            cost = time() - start # stop timer (cost estimate)
            acc = np.mean(scores == y) 
            RES.append([suj,cl,acc,cost])
            # print(suj, cl, str(round(acc * 100, 2))+'%', str(int(cost*1000))+'ms')
    return np.array(RES)



def sbcsp(args):
    filtro, order, fl, fh, t_start, t_end, n_comp, n_bands, Clog, path = args
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    bin0 = int(fl * (fs/nf))
    binN = int(fh * (fs/nf))
    RES = []
    for cl in classes:
        for suj in subjects:   
            XT = np.load(path + str(suj) + 'T.npy')
            XV = np.load(path + str(suj) + 'E.npy')
            
            XT = ([XT[cl[0]-1], XT[cl[1]-1]])
            XV = ([XV[cl[0]-1], XV[cl[1]-1]])
            
            XT = np.concatenate([XT[0],XT[1]]) # Dados de treinamento das classes A, B
            XV = np.concatenate([XV[0],XV[1]]) # Dados de validação das classes A, B
            y = np.concatenate([np.zeros(int(len(XT)/2)), np.ones(int(len(XT)/2))]) # Vetor gabarito
            
            XTJ = XT[:,:,W_Start:W_End] 
            XVJ = XV[:,:,W_Start:W_End]
            
            start = time() # start timer
            
            if filtro == 'FFT':
                # Spectral filtering
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
            if filtro == 'FFT': n_bins = len(XT_FFT[0,0,:]) # ou (fh-fl) * 4 # Número total de bins de frequencia 
            elif filtro == 'IIR': n_bins = fh-fl 
            overlap = 2
            step = int(n_bins / n_bands)
            size = int(step * overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
            
            # Make sub-bands limits and Temporal/Spectral filtering
            SCORE_T = np.zeros((len(XT), n_bands))
            SCORE_V = np.zeros((len(XV), n_bands))
            for i in range(n_bands):
                if filtro == 'FFT':
                    # Only Spectral filtering
                    bin_ini = i*step
                    bin_fim = i*step+size
                    if bin_fim >= n_bins: bin_fim = n_bins-1
                    XTF = XT_FFT[:, :, bin_ini:bin_fim]
                    XVF = XV_FFT[:, :, bin_ini:bin_fim]
                elif filtro == 'IIR':
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
                    XTF = lfilter(b, a, XTJ) 
                    XVF = lfilter(b, a, XVJ)
                
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
        
            #print(suj, cl, str(round(acc*100,2))+'%', str(int(cost*1000))+'ms')
            RES.append([suj,cl,acc,cost]) 
    
    return np.asarray(RES)


if __name__ == "__main__":
    path = '/mnt/dados/eeg_data/eeg_epochs/BCI4_2a/A0'
    filtro = 'FFT' #'IIR' or 'FFT'
    order = 5
    fl = 0
    fh = 40
    n_comp = 2
    n_bands = 15
    Clog = -3
    t_start,t_end = 2.5,4.5
    args = (filtro, order, fl, fh, t_start, t_end, n_comp, n_bands, Clog, path)
    RES = sbcsp(args) #csp_lda or sbcsp 
    print(f'{str(round(RES[:,2].mean()*100,2))}%  +- {str(round(RES[:,2].std()*100,1))}% || {round(RES[:,3].sum(),1)}s \n\n')
    
    
    # RES1 = RES[:,:3]
    # RES1[:,2] = [round(RES[i,2]*100,2) for i in range(len(RES1))]
    
    # ACC = RES1[:,2].reshape(6,9)
    
    # [ print(round(ACC[i,:].mean(),2), round(ACC[i,:].std(),1)) for i in range(0,6) ]
    
    # [ print(round(ACC[:,j].mean(),2), round(ACC[:,j].std(),1)) for j in range(0,9) ]
    
    # plt.figure(figsize=(10, 7), facecolor='mintcream')
    # plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    # plt.boxplot(ACC, vert = True, showfliers = True, notch = False, patch_artist = True, 
    #             boxprops=dict(facecolor="lavender", color="gray", linewidth=1, hatch = ''))
    # plt.xlabel('Subject', size=14)
    # plt.ylabel('Accuracy (%)', size=14)
    # plt.yticks(np.arange(50, 100, step=5))
    # plt.xticks(np.arange(1, 10, step=1))
    # # plt.title('Boxplot: Acurácia do classificador LDA por sujeito (MD x ME) - 28 janelas x 9 sujeitos')
    # plt.savefig('boxplot_subj_SBCSP_IIR_0-40.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    # plt.figure(figsize=(10, 7), facecolor='mintcream')
    # plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    # plt.boxplot(ACC.T, vert = True, showfliers = True, notch = False, patch_artist = True, 
    #             boxprops=dict(facecolor="lavender", color="gray", linewidth=1, hatch = ''))
    # plt.xlabel('Task', size=14)
    # plt.ylabel('Accuracy (%)', size=14)
    # plt.yticks(np.arange(50, 100, step=5))
    # plt.xticks(np.arange(0, 7), ('','LR', 'LF', 'LT', 'RF', 'RT', 'FT'))
    # # plt.title('Boxplot: Acurácia do classificador LDA por sujeito (MD x ME) - 28 janelas x 9 sujeitos')
    # plt.savefig('boxplot_tasks_SBCSP_IIR_0-40.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
        