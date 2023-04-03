# -*- coding: utf-8 -*-
from numpy import load, transpose, asarray, concatenate, zeros, ones, unique, dot, empty, arange, mean, log, real, imag, ravel, std
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.fftpack import fft
import itertools
from sklearn.svm import SVC
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy import stats


class CSP():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        e, c, t = X.shape
        classes = unique(y)
        X0 = X[classes[0] == y,:,:]
        X1 = X[classes[1] == y,:,:]
        # Sum up covariance matrix
        S0 = zeros((c, c))
        S1 = zeros((c, c))
        for i in range(int(e/2)): # add conversão int() ?
            S0 += dot(X0[i,:,:], X0[i,:,:].T)
            S1 += dot(X1[i,:,:], X1[i,:,:].T)
        [D, W] = eigh(S0, S0 + S1)
        ind = empty(c, dtype=int)
        ind[0::2] = arange(c - 1, c // 2 - 1, -1)
        ind[1::2] = arange(0, c // 2)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]

    def transform(self, X):
        XT = asarray([dot(self.filters_, epoch) for epoch in X])
        XVAR = log(mean(XT ** 2, axis=2))
        return XVAR
    
    
if __name__ == "__main__":
    
    X = [] # vetor de dimensão 4 onde as duas primeiras contém os dados de treinamento das duas classes e as duas últimas os dados de validação das duas classes      
    X.append(load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01T_1.npy', 'rb')))
    X.append(load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01T_2.npy', 'rb')))
    XT = concatenate([X[0],X[1]]) 
    XJ = XT[:,:,125:625]
    XF = fft(XJ)
    REAL = transpose(real(XF)[:,:,:80], (2, 0, 1)) #transpoe para intercalar
    IMAG = transpose(imag(XF)[:,:,:80], (2, 0, 1)) #transpoe para intercalar
    XF = list(itertools.chain.from_iterable(zip(IMAG, REAL))) #intercalando
    XF = transpose(XF, (1, 2, 0)) # retorna ao formato original      
    
    
    
    # Divide sub-bands
    n_bins = len(list(XF[0][0]))
    overlap = 2
    n_bandas = 33
    step = int((n_bins / n_bandas / 2) * 2)
    size = int(step * overlap)
    while True:
        last_end = (n_bandas-1) * step + size
        if last_end <= n_bins:
            break
        step -= 2
        
    
    XT = [XT[:, :, i*step:i*step+size] for i in range(n_bandas)] # [ 33 x 144 x 22 x 8 ]
    # XV = [XV[:, :, i*step:i*step+size] for i in range(n_bandas)] # [ 33 x 144 x 22 x 8 ]
    
    
    y = concatenate([zeros(72), ones(72)]) #vetor gabarito
    
    # CSP
    csp = [CSP(n_components=4) for i in range(n_bandas)]

    for i in range(n_bandas): csp[i].fit(XT[i], y)

    XT_CSP = [csp[i].transform(XT[i]) for i in range(n_bandas)]
    # XV_CSP = [csp[i].transform(XV[i]) for i in range(n_bandas)]
    
    X1 = XF
    X2 = list(XT)
    X3 = list(XT_CSP)
    
    # LDA
    SCORE_T = zeros((144, n_bandas))
    SCORE_T2 = zeros((144, n_bandas))
    # SCORE_V = zeros((144, n_bandas))
    clf = [LinearDiscriminantAnalysis() for i in range(n_bandas)]
    for i in range(n_bandas):
        clf[i].fit(XT_CSP[i], y)
        SCORE_T2[:, 1] = ravel(clf[i].predict(XT_CSP[i]))
        SCORE_T[:, i] = ravel(clf[i].transform(XT_CSP[i])) # classificaçoes de cada época nas N sub bandas - auto validação
        # SCORE_V[:, i] = ravel(clf[i].transform(XV_CSP[i])) # validação
        
    a = SCORE_T
    d = SCORE_T2
        
    # Meta-classificador Bayesiano
    SCORE_T0 = SCORE_T[y == 0, :]
    SCORE_T1 = SCORE_T[y == 1, :]
    
    aa = SCORE_T0
    aaa = SCORE_T1
    
    m0 = mean(SCORE_T0, axis=0) #media classe A
    std0 = std(SCORE_T0, axis=0) #desvio padrão classe A
    
    m1 = mean(SCORE_T1, axis=0)
    std1 = std(SCORE_T1, axis=0)

    p0 = norm(m0, std0) # p0 e p1 representam uma distribuição normal de médias m0 e m1, e desvio padrão std0 e std1
    p1 = norm(m1, std1)
    
    stats.probplot(p1.rvs(), plot = plt)
    
    META_SCORE_T = log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
    
    bb = META_SCORE_T
    # META_SCORE_V = log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
    
    # SVM on top of the meta-classifier
    svc = SVC(kernel="linear", C=10**(-4))
    svc.fit(META_SCORE_T, y)
    # ans = svc.predict(META_SCORE_V)
    
    # acuracia = mean(ans == y)
    
    # print(sujeito, classes, round(acuracia * 100, 2))     