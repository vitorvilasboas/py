# -*- coding: utf-8 -*-
from numpy import load, transpose, asarray, concatenate, zeros, ones
from numpy import unique, dot, empty, arange, mean, log, real, imag, std
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import lfilter, butter
from time import time
from scipy.fftpack import fft
import itertools


def loading(sujeito, classes):
    XT = []  # vetor de dimensão 3 onde as duas primeiras contém os dados de treinamento das duas classes e as duas últimas os dados de validação das duas classes      
    for j in range(2):
        path = '/home/vboas/devto/datasets/BCICIV_2a/npy/A0' + str(sujeito) + 'T_' + str(classes[j]) + '.npy'
        dados = load(open(path, 'rb'))
        XT.append(dados)
    
    XV = []  # vetor de dimensão 4 onde as duas primeiras contém os dados de treinamento das duas classes e as duas últimas os dados de validação das duas classes      
    for j in range(2):
        path = '/home/vboas/devto/datasets/BCICIV_2a/npy/A0' + str(sujeito) + 'E_' + str(classes[j]) + '.npy'
        dados = load(open(path, 'rb'))
        XT.append(dados)
            
    return XT,XV

def windowing(X, fs, t_start, t_end, atraso):
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    XJ = []
    atraso = atraso
    for i in range(4):

        janela = transpose(X[i][:,:,W_Start:W_End], (1, 0, 2))
        for cont in range(1, atraso + 1):
            jAtrasada = transpose(X[i][:,:,W_Start-cont:W_End-cont], (1, 0, 2))
            janela = concatenate([janela, jAtrasada])

        XJ.append(transpose(janela, (1, 0, 2)))
    return XJ

def filteringFFT(X, fs, fl, fh):
    nf = fs/2.
    bin0 = int(fl * (fs/nf))  # para fl = 8 bin0 = 15
    binN = int(fh * (fs/nf)) # para fl = 8 bin0 = 15
    # print bin0, binN
    XF = []
    for i in range(4): # filtra os dados de treinamento e validação das duas classes 
        filtrado = fft(X[i])
        REAL = transpose(real(filtrado)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
        IMAG = transpose(imag(filtrado)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
        filtrado = list(itertools.chain.from_iterable(zip(IMAG, REAL))) #intercalando
        filtrado = transpose(filtrado, (1, 2, 0)) # retorna ao formato original      
        XF.append(filtrado)
    return XF

def filteringIIR(X, fs, fl, fh):
    nf = fs/2.
    b, a = butter(ordem, [fl/nf, fh/nf], btype='bandpass')
    XF = []
    for i in range(4): # filtra os dados de treinamento e validação das duas classes 
        filtrado = lfilter(b, a, X[i])
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
    
def processing(sujeito, classes, args):
    atraso, t_ini, t_end, filtro, n_comp, fs, fl, fh, ordem = args  
    
    XT,XV = loading(sujeito, classes) # Loading dataset
    
    start = time() # start timer
    
    XTJ = windowing(XT, fs, t_start, t_end, atraso) # Windowing
    XVJ = windowing(XT, fs, t_start, t_end, atraso) 
    
    if filtro == 'IIR': 
        XTF = filteringIIR(XTJ, fs, fl, fh) # Temporal/Spectral filtering
        XVF = filteringIIR(XVJ, fs, fl, fh)
    elif filtro == 'FFT': 
        XTF = filteringFFT(XTJ, fs, fl, fh)
        XVF = filteringFFT(XTJ, fs, fl, fh)
    
    # Concatenating training and evaluating data
    #XT = concatenate([XF[0],XF[1]]) # Dados de treinamento das classes A, B
    #XV = concatenate([XF[2],XF[3]]) # Dados de validação das classes A, B
    y = concatenate([zeros(72), ones(72)]) # Criando vetor gabarito
    # CSP
    csp = CSP(n_components=n_comp)
    csp.fit(XTF, y)
    XT_CSP = csp.transform(XTF)
    XV_CSP = csp.transform(XVF) 
    # LDA
    clf = LinearDiscriminantAnalysis()
    clf.fit(XT_CSP, y)
    saida_lda = clf.predict(XV_CSP)
    # Results
    acuracia = mean(saida_lda == y)
    tempo = time() - start # stop timer (cost estimate)
    # print(sujeito, classes, round(tempo, 2), round(acuracia * 100, 2))
    return acuracia, tempo
#
if __name__ == "__main__":
    sujeitos = arange(1, 2)
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    atrasos = range(0,6)
    t_start,t_end = 2.5,4.5
    filtro = ['IIR']  # 'IIR' 'FFT'
    n_comp = 6 # componentes CSP (3 pares = 6)
    fs = 250
    fl = 8 
    fh = 30
    ordem = 5
    
    for ft in filtro:
        print(ft + ': Atrasos ','Acc Média ','DP ','Custo ','Custo_i')
        for atraso in atrasos:
            
            args = atraso, t_start, t_end, ft, n_comp, fs, fl, fh, ordem
            result = [processing(suj, cl, args) for suj in sujeitos for cl in classes ]
            
            acuracias = asarray(result)[:,0]
            tempos = asarray(result)[:,1]
            
            acc_media = round(mean(acuracias)*100, 1)
            acc_dsvpadrao = round(std(acuracias)*100, 1)
            
            custo_total = round(sum(tempos),1)
            custo_medio = int(mean(tempos)*1000)
            
            print (str(atraso), str(acc_media)+'%', str(acc_dsvpadrao)+'%', str(custo_total)+'s', str(custo_medio)+'ms')            
            