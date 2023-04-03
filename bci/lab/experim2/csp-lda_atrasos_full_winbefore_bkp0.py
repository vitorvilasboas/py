# -*- coding: utf-8 -*-
from numpy import load, transpose, dtype, fromfile, asarray, concatenate, zeros, ones
from numpy import unique, dot, empty, arange, mean, log, real, imag, ravel, std, cos, pi, sin
from scipy.linalg import eigh, pinv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import filtfilt, lfilter, butter
from time import time
from scipy.fftpack import fft
import itertools
from sklearn.svm import SVC
from scipy.stats import norm

def load_data(sujeito, classes):
    ds = ['T_', 'E_']
    X = []  # vetor de dimensão 4 onde as duas primeiras contém os dados de treinamento das duas classes e as duas últimas os dados de validação das duas classes      
    for i in range(2):
        for j in range(2):
            path = '/home/vboas/devto/datasets/BCICIV_2a/npy/A0' + str(sujeito) + ds[i] + str(classes[j]) + '.npy'
            dados = load(open(path, 'rb'))
            X.append(dados)
    return X

def windowing(X, fs, t_0, t_start, t_end, atraso):
    W_Start = int((t_start - t_0) * fs)
    W_End = int((t_end - t_0) * fs)
    XJ = []
    atraso = atraso
    for i in range(4):
        # XJ.append(XF[i][:, :, W_Start:W_End])
        janela = transpose(X[i][:,:,W_Start:W_End], (1, 0, 2))
        for cont in range(1, atraso + 1):
            jAtrasada = transpose(X[i][:,:,W_Start-cont:W_End-cont], (1, 0, 2))
            janela = concatenate([janela, jAtrasada])
        #print(janela[0][0][0],' é igual a ',janela[22][0][1])
        XJ.append(transpose(janela, (1, 0, 2)))
    #print(XJ[0][0][0][498],' é igual a ',XJ[0][0][22][499])
    return XJ

def filtragemFFT2(XJ, fs, fl, fh):
    nf = fs/2.
    bin0 = int(fl * (fs/nf))  # para fl = 8 bin0 = 15
    binN = int(fh * (fs/nf)) # para fl = 8 bin0 = 15
    # print bin0, binN
    XF = []
    for i in range(4): # filtra os dados de treinamento e validação das duas classes 
        filtrado = fft(XJ[i])
        REAL = transpose(real(filtrado)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
        IMAG = transpose(imag(filtrado)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
        filtrado = list(itertools.chain.from_iterable(zip(IMAG, REAL))) #intercalando
        filtrado = transpose(filtrado, (1, 2, 0)) # retorna ao formato original      
        XF.append(filtrado)
    return XF

def filtragemIIR(XJ, fs, fl, fh):
    nf = fs/2.
    b, a = butter(ordem, [fl/nf, fh/nf], btype='bandpass')
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
    
def sbcsp(sujeito, classes, args):
    atraso, t_filtro, n_componentes, fs, fl, fh, ordem = args  
    
    X = load_data(sujeito, classes)
    
    t0 = time()
    
    XJ = windowing(X, fs, 0, 2.5, 4.5, atraso)
    
    if t_filtro == 'IIR': XF = filtragemIIR(XJ, fs, fl, fh)
    elif t_filtro == 'FFT': XF = filtragemFFT2(XJ, fs, fl, fh)

    XT = concatenate([XF[0],XF[1]]) # treinamento A e B
    XV = concatenate([XF[2],XF[3]]) # validação A e B
    
    y = concatenate([zeros(72), ones(72)]) #vetor gabarito
    
    # CSP
    csp = CSP(n_components=n_componentes)
    csp.fit(XT, y)
    XT_CSP = csp.transform(XT)
    XV_CSP = csp.transform(XV) 
    
    # LDA
    clf = LinearDiscriminantAnalysis()
    clf.fit(XT_CSP, y)
    saida_lda = clf.predict(XV_CSP)
    
    # for aa in saida_lda: print(aa)
    
    acuracia = mean(saida_lda == y)
    
    tempo = time() - t0
    
    print(sujeito, classes, round(tempo, 2), round(acuracia * 100, 2))
    #print(round(acuracia * 100, 2))
    # print asarray(XJ).shape 
    
    return acuracia, tempo

if __name__ == "__main__":
    # sujeito = arange(1, 10)
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    atrasos = range(0,6) 
    t_filtro = ['IIR']  # opções: 'IIR' 'FFT' 'PROJ'
    n_componentes = 6
    fs = 250
    fl = 8 
    fh = 30
    ordem = 5
    
    print('\n>> Qtd sujeitos: 9')
    print('>> Classes: LH x RH')
    print('>> Abordagem: CSP + LDA')
    print('>> Parâmetros: [8-30Hz] butter_order=5 ')
    
    for tf in t_filtro:
        print('\n========================',tf,'=========================')
        print('Atrasos\t','| Acc Média (%)','| DP (%)', '| Custo (s)   ', '| Custo_i (s)')
        print('======================================================')
        for atraso in atrasos:
            args = atraso, tf, n_componentes, fs, fl, fh, ordem
            result = [sbcsp(suj, classes[i], args) for suj in range(1, 2) for i in range(0, 1) ]
            print ('  '+ str(atraso) +'\t |  ' + 
                   str(round(mean(asarray(result)[:,0])*100, 2)) + '%\t |  ' +
                   str(round(std(asarray(result)[:,0])*100, 2)) + '% |  ' +
                   str(round(sum(asarray(result)[:,1]), 2)) + 's\t| ' +
                   str(round(mean(asarray(result)[:,1]), 2)) + 's')
            
            #print ('Acc média:\t' + str(round(mean(asarray(result)[:,0])*100, 2)) + '%')
            #print ('Custo total:\t' + str(round(sum(asarray(result)[:,1]), 2)) + 's')
            #print ('Custo médio:\t' + str(round(mean(asarray(result)[:,1]), 2)) + 's\n')