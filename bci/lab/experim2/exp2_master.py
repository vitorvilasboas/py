# -*- coding: utf-8 -*-
from numpy import load, transpose, asarray, concatenate, zeros, ones
from numpy import unique, dot, empty, arange, mean, log, real, imag, std
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import lfilter, butter
from scipy.fftpack import fft
import itertools

class CSP():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        e, c, t = X.shape
        classes = unique(y)        
        X0 = X[classes[0] == y,:,:]
        X1 = X[classes[1] == y,:,:]
        S0 = zeros((c, c)) # Sum up covariance matrix
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
    sujeitos = range(1,10)
    pares_classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    atrasos = range(0,6)
    n_comp = 6 # componentes CSP (3 pares = 6)
    fl = 8 
    fh = 30
    fs = 250
    nf = fs/2.
    filtro = 'IIR'
    winStart = int(2.5 * fs)
    winEnd = int(4.5 * fs)
    
    FINAL = []
    for ats in atrasos:
        ACCs = zeros([len(sujeitos),len(pares_classes)])
        for suj in sujeitos:
            for classes,cl in zip(pares_classes,range(len(pares_classes))):
                # Loading dataset
                XT = []
                XV = [] 
                for j in range(2):
                    path = '/home/vboas/devto/datasets/BCICIV_2a/npy/A0' + str(suj) + 'T_' + str(classes[j]) + '.npy'
                    dados = load(open(path, 'rb'))
                    XT.append(dados)
                    path = '/home/vboas/devto/datasets/BCICIV_2a/npy/A0' + str(suj) + 'E_' + str(classes[j]) + '.npy'
                    dados = load(open(path, 'rb'))
                    XV.append(dados)
                
                # Windowing
                XTJ = []
                XVJ = []
                for i in range(len(XT)):
                    janelaT = XT[i][:,:,winStart:winEnd]
                    janelaV = XV[i][:,:,winStart:winEnd]
                    for cont in range(1, ats + 1):
                        jAtrasadaT = XT[i][:,:,winStart-cont:winEnd-cont]
                        janelaT = concatenate([janelaT, jAtrasadaT], axis=1)
                        jAtrasadaV = XV[i][:,:,winStart-cont:winEnd-cont]
                        janelaV = concatenate([janelaV, jAtrasadaV], axis=1)
                    XTJ.append(janelaT)
                    XVJ.append(janelaV)
                
                # Temporal/Spectral filtering
                XTF = []
                XVF = []
                if filtro == 'IIR':
                    b, a = butter(5, [fl/nf, fh/nf], btype='bandpass') # to filt IIR
                    for i in range(len(XTJ)): # filtra os dados de treinamento e validação das duas classes 
                        XTF.append( lfilter(b, a, XTJ[i]) )
                        XVF.append( lfilter(b, a, XVJ[i]) )   
                elif filtro == 'FFT':
                    bin0 = int(fl * (fs/nf)) # para fl = 8 bin0 = 15 # to FFT
                    binN = int(fh * (fs/nf)) # para fl = 8 bin0 = 15 # to FFT
                    for i in range(len(XTJ)): # filtra os dados de treinamento e validação das duas classes 
                        filtrado = fft(XTJ[i])
                        REAL = transpose(real(filtrado)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
                        IMAG = transpose(imag(filtrado)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
                        filtrado = list(itertools.chain.from_iterable(zip(IMAG, REAL))) #intercalando
                        filtrado = transpose(filtrado, (1, 2, 0)) # retorna ao formato original      
                        XTF.append(filtrado)
                        
                        filtrado = fft(XVJ[i])
                        REAL = transpose(real(filtrado)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
                        IMAG = transpose(imag(filtrado)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
                        filtrado = list(itertools.chain.from_iterable(zip(IMAG, REAL))) #intercalando
                        filtrado = transpose(filtrado, (1, 2, 0)) # retorna ao formato original      
                        XVF.append(filtrado)
                
                # Concatenating training and evaluating data
                XTF = concatenate([XTF[0],XTF[1]]) # Dados de treinamento das classes A, B
                XVF = concatenate([XVF[0],XVF[1]]) # Dados de validação das classes A, B
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
                
                acc = mean(saida_lda == y) # Results
                #print (sujeito, classes, str(atraso), str(round(acuracia * 100, 2))+'%')
                
                ACCs[suj-1,cl] = acc

        FINAL.append(ACCs)
        
    MEANS = [ [(str(round(FINAL[ats][suj-1,:].mean()*100,1)) + str(" ~ ") + 
                        str(round(FINAL[ats][suj-1,:].std()*100,1))) 
                        for suj in sujeitos ] for ats in atrasos ]
    
    [ MEANS[ats].append(str(round(mean(FINAL[ats])*100,1)) + " ~ " +
                    str(round(std(FINAL[ats])*100,1))) for ats in atrasos]
    
    MEANS = asarray(MEANS).T # [suj,ats] com média final por atraso na última linha
    
    