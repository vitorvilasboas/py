# -*- coding: utf-8 -*-
from numpy import load, transpose, asarray, concatenate, zeros, ones, unique, dot, empty, arange, mean, log, where
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import lfilter, butter
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

def carrega(classes):
    ds = ['T_', 'E_']
    X = []  # vetor de dimensão 4 onde as duas primeiras contém os dados de treinamento das duas classes e as duas últimas os dados de validação das duas classes      
    for i in range(2):
        for j in range(2):
            path = '/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01' + ds[i] + str(classes[j]) + '.npy'
            dados = load(open(path, 'rb'))
            X.append(dados)
    return X

def janela(X, fs=250., atraso=0):
    W_Start = int(0.5 * fs)
    W_End = int(2.5 * fs)
    XJ = []
    atraso = atraso
    for i in range(4):
        janela = transpose(X[i][:,:,W_Start:W_End], (1, 0, 2))
        for cont in range(1, atraso + 1):
            jAtrasada = transpose(X[i][:,:,W_Start-cont:W_End-cont], (1, 0, 2))
            janela = concatenate([janela, jAtrasada])
        XJ.append(transpose(janela, (1, 0, 2)))
    return XJ

def filtra(XJ, fs=250., fl=8., fh=30.):
    nf = fs/2.
    b, a = butter(5, [fl/nf, fh/nf], btype='bandpass')
    XF = []
    for i in range(4): # filtra os dados de treinamento e validação das duas classes 
        filtrado = lfilter(b, a, XJ[i])
        XF.append(filtrado)
    return XF

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
    classes = [[1, 2]]
    X = carrega(classes[0])
    XJ = janela(X)
    XF = filtra(XJ)
    XT = concatenate([XF[0],XF[1]]) # treinamento A e B
    XV = concatenate([XF[2],XF[3]]) # validação A e B
    y = concatenate([zeros(72), ones(72)])
    # CSP
    csp = CSP(n_components=6)
    csp.fit(XT, y)
    XT_CSP = csp.transform(XT)
    XV_CSP = csp.transform(XV) 
    
    # LDA
    clf = LinearDiscriminantAnalysis()
    clf.fit(XT_CSP, y)
    saida_lda = clf.predict(XV_CSP)
    acuracia = mean(saida_lda == y)
    
    #idx1 = where(y == 0)[0]
    #idx2 = where(y == 1)[0]
    CM = confusion_matrix(y, saida_lda)
    print('Acurácia: ', round(acuracia * 100, 2),'%')
    print(CM)
    
    redeNeural = MLPClassifier(verbose=False, max_iter=10000, activation='logistic', learning_rate_init=0.0001)
    redeNeural.fit(XT_CSP, y)
    saida_rna = redeNeural.predict(XV_CSP)
    acuracia2 = mean(saida_rna == y)
    
    CM = confusion_matrix(y, saida_rna)
    print('\nAcurácia: ', round(acuracia2 * 100, 2),'%')
    print(CM)
    