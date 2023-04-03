# -*- coding: utf-8 -*-
from numpy import load, transpose, dtype, fromfile, asarray, concatenate, zeros, ones
from numpy import unique, dot, empty, arange, mean, log
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import filtfilt, lfilter, butter
import matplotlib.pyplot as plt
from time import time

def csp_lda(sujeito, classe, atraso, n_componentes, fl, fh, ordem):         
            
    ds = ['T_', 'E_']
    X = []  # vetor de dimensão 4 onde as duas primeiras contém os dados de treinamento das duas classes e as duas últimas os dados de validação das duas classes      
    for i in range(2):
        for j in range(2):
            path = '/mnt/dados/bci_tools/datasets/bci_cIV_2a/npy/epocas_t2/A0' + str(sujeito) + ds[i] + str(classe[j]) + '.npy'
            dados = load(open(path, 'rb'))
            X.append(dados)
    
    ## Janelamento
    t0 = time()
    fs = 250
    W_Start = int(0.5*fs)
    W_End = int(2.5*fs)
    XJ = []
    atraso = atraso
    for i in range(4):
        # XJ.append(XF[i][:, :, W_Start:W_End])
        janela = transpose(X[i][:,:,W_Start:W_End], (1, 0, 2))    
        for cont in range(1, atraso + 1):
            jAtrasada = transpose(X[i][:,:,W_Start-cont:W_End-cont], (1, 0, 2))
            janela = concatenate([janela, jAtrasada])
        XJ.append(transpose(janela, (1, 0, 2)))
    
    ## Filtragem
    nf = fs/2.
    b, a = butter(ordem, [fl/nf, fh/nf], btype='bandpass')
    XF = []
    for i in range(4): # filtra os dados de treinamento e validação das duas classes 
        filtrado = lfilter(b, a, XJ[i])
        # filtrado = filtfilt(b, a, X[i])
        XF.append(filtrado)
    
    
    XT = concatenate([XF[0],XF[1]]) # treinamento A e B
    XV = concatenate([XF[2],XF[3]]) # validação A e B
    
    y = concatenate([zeros(72), ones(72)]) #vetor gabarito
    
    # CSP - EXTRAÇÃO DE CARACTERÍSTICAS
    e, c, t = XT.shape
    classes = unique(y)
    
    X0 = XT[classes[0] == y,:,:]
    X1 = XT[classes[1] == y,:,:]
    
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
    
    filters_ = W.T[:n_componentes]
    
    XTtrans = asarray([dot(filters_, epoch) for epoch in XT])
    XVtrans = asarray([dot(filters_, epoch) for epoch in XV])
    
    XT_CSP = log(mean(XTtrans ** 2, axis=2))
    XV_CSP = log(mean(XVtrans ** 2, axis=2))
    
    ### CLASSIFICAÇÃO (LDA)
    clf = LinearDiscriminantAnalysis() #INSTANCIA A CLASSE LDA
    clf.fit(XT_CSP, y)
    
    acuracia = mean(clf.predict(XV_CSP) == y)
    tempo = time() - t0
    #print(sujeito, classe, round(tempo, 2), round(acuracia * 100, 2))
    print(round(acuracia * 100, 2))
    # print(asarray(XJ).shape) 
    
    return acuracia, tempo
    
    
if __name__ == "__main__":
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    # sujeito = 5
    atraso = 0 
    n_componetes = 6
    fl = 8 
    fh = 30
    ordem = 5
    result = [csp_lda(suj, classes[i], atraso, n_componetes, fl, fh, ordem) for suj in range(1, 10) for i in range(0, 1) ]
    #print('Acc média: ' + str(round(mean(asarray(result)[:,0])*100, 2)) + '%')
    #print('Custo total: ' + str(round(sum(asarray(result)[:,1]), 2)) + 's')
    #print('Custo médio: ' + str(round(mean(asarray(result)[:,1]), 2)) + 's')
    
    print(str(round(mean(asarray(result)[:,0])*100, 2)))



