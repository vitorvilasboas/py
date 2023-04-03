# -*- coding: utf-8 -*-
from numpy import transpose, dtype, fromfile, asarray, concatenate, zeros, ones
from numpy import unique, dot, empty, arange, mean, log, real, imag, cos, pi, sin
from scipy.linalg import eigh, pinv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from time import time

def csp_lda(sujeito, classe, atraso, n_componentes, fl, fh, ordem):
    ds = ['T_', 'E_']
    X = []  # vetor de dimensão 4 onde as duas primeiras contém os dados de treinamento das duas classes e as duas últimas os dados de validação das duas classes      
    for i in range(2):
        for j in range(2):
            path = '/mnt/dados/bci_tools/dset42a/fdt/epocas_t2/A0' + str(sujeito) + ds[i] + str(classe[j]) + '.fdt'
            fid = open(path, 'rb')
            dados = fromfile(fid, dtype('f'))
            dados = dados.reshape((72, 1000, 22)) #Dá uma nova forma a uma matriz sem alterar seus dados
            dados = transpose(dados, (0,2,1))
            X.append(dados)
            # X.append('A0' + str(sujeito) + ds[i] + str(classe[j]) + '.fdt')
    
    t0 = time()
    fs = 250
   
    ## Janelamento
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
    XF = []
    for j in range(4): # filtra os dados de treinamento e validação das duas classes
        XJi = XJ[j]
        ne = XJi.shape[0] # n_epocas
        na = XJi.shape[2] # n_amostras
        fs=250
        nf = fs/2.
        bin0 = int(fl * (fs/nf) - 1)  # para fl = 8 bin0 = 15
        binN = int(fh * (fs/nf) - 1) # para fh = 30 bin0 = 59
        sizeBase = binN - bin0
        fi = fl/float(fs) # converte para float (evita from __future__ import division)
        fm = fh/float(fs) 
        t = arange(na)
        B = zeros((na, 2*sizeBase)) #criamos uma base de senos e cossenos, de dimensão (nt, 2*m)
        for i in range(sizeBase):
            f = fi + i / float(sizeBase) * (fm-fi) # Calculamos a frequencia reduzida correspondente ao bin i
            b_sin = sin(2*pi*f*t) # Obtemos o seno e o cosseno dessa frequencia
            b_cos = cos(2*pi*f*t)
            B[:, 2*i] = b_sin # Colocamos os dois nas posições 2*i e 2*i+1 da base
            B[:, 2*i+1] = b_cos # [Seno_0, Cosseno_0, Seno_1, Cosseno_1, etc]
        # G0 é a matriz de projeção, que liga diretamente o sinal original e o sinal projetado na base de interesse
        G0 = dot(pinv(dot(B.T, B)), B.T)
        # Para cada época do vetor X fazemos esta projeção, multiplicando a matriz X[k,:,:] por G0
        XFi = asarray([dot(XJi[k, :, :], G0.T) for k in range(ne)])
        XF.append(XFi)
    
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
    # print(sujeito, classe, round(tempo, 2), round(acuracia * 100, 2))
    
    # print asarray(XJ).shape 
    
    return acuracia, tempo
    
    
if __name__ == "__main__":
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    # sujeito = 5
    atraso = 0 
    n_componetes = 6
    fl = 8 
    fh = 30
    ordem = 5
    result = [csp_lda(suj, classes[i], atraso, n_componetes, fl, fh, ordem) for suj in range(1, 10) for i in range(0, 6) ]
    print('Acc média: ' + str(round(mean(asarray(result)[:,0])*100, 2)) + '%')
    print('Custo total: ' + str(round(sum(asarray(result)[:,1]), 2)) + 's')
    print('Custo médio: ' + str(round(mean(asarray(result)[:,1]), 2)) + 's')





