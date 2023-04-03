# -*- coding: utf-8 -*-
from numpy import load, transpose, asarray, concatenate, zeros, ones, array
from numpy import unique, dot, empty, arange, mean, log, real, imag, std, ravel
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import lfilter, butter
from time import time
from scipy.fftpack import fft
import itertools
from scipy.stats import norm
from sklearn.svm import SVC

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
        for i in range(int(e/2)): 
            S0 += dot(X0[i,:,:], X0[i,:,:].T)   # classe A
            S1 += dot(X1[i,:,:], X1[i,:,:].T)   # classe B

        [D, W] = eigh(S0, S0 + S1)

        ind = empty(c, dtype=int) # usado para a ideia de pares de componentes (extremos juntos)
        ind[0::2] = arange(c - 1, c // 2 - 1, -1)
        ind[1::2] = arange(0, c // 2) #ind [21 0 20 1 19 2 ... 12 9 11 10]
        
        W1 = W[:, ind] # reorganizando W conforme ind 
        
        self.filters_ = W1.T[:self.n_components] # considera somente as n_components colunas de W1 que passam a ser linhas com a transposição

    def transform(self, X):
        XT = asarray([dot(self.filters_, epoch) for epoch in X]) #epoch = [22,4]
        XVAR = log(mean(XT ** 2, axis=2))
        
        return XVAR
    
if __name__ == "__main__":
    
    sujeitos = range(1,10)
    pares_classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    n_comp = 6 # componentes CSP (3 pares = 6)
    fs = 250
    Clog = -4
    
    filtro = 'IIR'
    fl = 0
    fh = 40
    ordem = 5
    nf = fs/2.
    #n_bandas = 33
    NBANDAS = range(1,fh+1)
    
    
    bin0 = int(fl * (fs/nf))  # para fl = 8 bin0 = 16 # to FFT
    binN = int(fh * (fs/nf)) # para fh = 30 bin0 = 60 # to FFT
    
    t_start,t_end = 2.5,4.5
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    
    acuracias = zeros([len(sujeitos),len(pares_classes)])
    tempos = zeros([len(sujeitos),len(pares_classes)])
    
    acc_medias_suj = zeros([len(sujeitos)])
    tempos_medios_suj = zeros([len(sujeitos)])
    acc_dvpadrao_suj = zeros([len(sujeitos)])
    tempos_dvpadrao_suj = zeros([len(sujeitos)])
    
    result_nb = []
    
    nb = []
    
    for n_bandas in NBANDAS:
        for sujeito in sujeitos:
            #print("\n")
            for classes,cl in zip(pares_classes,range(len(pares_classes))):
                #sujeito = 1
                #classes = [1,2]    
                # Loading dataset
                XT = []        
                for j in range(2):
                    path = '/home/vboas/devto/datasets/BCICIV_2a/npy/A0' + str(sujeito) + 'T_' + str(classes[j]) + '.npy'
                    dados = load(open(path, 'rb'))
                    XT.append(dados)
                
                XV = []  
                for j in range(2):
                    path = '/home/vboas/devto/datasets/BCICIV_2a/npy/A0' + str(sujeito) + 'E_' + str(classes[j]) + '.npy'
                    dados = load(open(path, 'rb'))
                    XV.append(dados)
                
                
                start = time() # start timer
                
                # Windowing
                XTJ = []
                XVJ = []
                for i in range(len(XT)):
                    janelaT = XT[i][:,:,W_Start:W_End]
                    janelaV = XV[i][:,:,W_Start:W_End]
                    XTJ.append(janelaT)
                    XVJ.append(janelaV)
                
                # Concatenating training and evaluating data
                XTC = concatenate([XTJ[0],XTJ[1]]) # Dados de treinamento das classes A, B [ ne*2 x nc x n_bins ]
                XVC = concatenate([XVJ[0],XVJ[1]]) # Dados de validação das classes A, B 
                y = concatenate([zeros(72), ones(72)]) # Criando vetor gabarito
                
                # Divide sub-bands
                n_bins = fh-fl
                overlap = 2
                step = int(n_bins / n_bandas)
                size = int(step * overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
                
                XTF = []
                XVF = [] 
                for i in range(n_bandas):
                    freq_low = i*step
                    freq_high = i*step+size
                    if freq_low == 0: freq_low = 0.001
                    if freq_high > fh: freq_high = fh
                    nb.append(array([freq_low,freq_high]))
            
                    b, a = butter(ordem, [freq_low/nf, freq_high/nf], btype='bandpass') # to filt IIR
                    XTF.append( lfilter(b, a, XTC) ) # Temporal/Spectral filtering
                    XVF.append( lfilter(b, a, XVC) ) # o filtro é aplicado por padrão na última dimensão
                
        
    
    # =============================================================================
    #             # Divide sub-bands
    #             n_bins = len(XTC[0,0,:])
    #             overlap = 2
    #             step = int(n_bins / n_bandas)
    #             while True: # regula
    #                 size = int(step * overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
    #                 last_end = (n_bandas-1) * step + size
    #                 if last_end <= n_bins:
    #                     break
    #                 print(last_end)
    #                 step -= 1 # default = 2 
    #             XTSB = [XTC[:, :, i*step:i*step+size] for i in range(n_bandas)] # [ n_band x ne*2 x nc x size ]
    #             XVSB = [XVC[:, :, i*step:i*step+size] for i in range(n_bandas)]
    #             
    # =============================================================================
                    # CSP
                csp = [CSP(n_components=n_comp) for i in range(n_bandas)]
                for i in range(n_bandas): csp[i].fit(XTF[i], y)
                XT_CSP = [csp[i].transform(XTF[i]) for i in range(n_bandas)]
                XV_CSP = [csp[i].transform(XVF[i]) for i in range(n_bandas)]
            
                # LDA
                SCORE_T = zeros((144, n_bandas))
                SCORE_V = zeros((144, n_bandas))
                clf = [LinearDiscriminantAnalysis() for i in range(n_bandas)]
                for i in range(n_bandas):
                    clf[i].fit(XT_CSP[i], y)
                    SCORE_T[:, i] = ravel(clf[i].transform(XT_CSP[i])) # classificaçoes de cada época nas N sub bandas - auto validação
                    SCORE_V[:, i] = ravel(clf[i].transform(XV_CSP[i])) # validação
                
                # Meta-classificador Bayesiano
                SCORE_T0 = SCORE_T[y == 0, :]
                m0 = mean(SCORE_T0, axis=0) #media classe A
                std0 = std(SCORE_T0, axis=0) #desvio padrão classe A
                
                SCORE_T1 = SCORE_T[y == 1, :]
                m1 = mean(SCORE_T1, axis=0)
                std1 = std(SCORE_T1, axis=0)
            
                p0 = norm(m0, std0) # p0 e p1 representam uma distribuição normal de médias m0 e m1, e desvio padrão std0 e std1
                p1 = norm(m1, std1)
                
                META_SCORE_T = log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
                META_SCORE_V = log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
                
                # SVM on top of the meta-classifier
                svc = SVC(kernel="linear", C=10**Clog)
                svc.fit(META_SCORE_T, y)
                saidas_svm = svc.predict(META_SCORE_V)
                
                acuracia = mean(saidas_svm == y) # Results
                tempo = time() - start # stop timer (cost estimate)
                
                #print (sujeito, classes, str(round(acuracia * 100, 2))+'%', str(int(tempo*1000))+'ms')
                
                acuracias[sujeito-1,cl] = acuracia
                tempos[sujeito-1,cl] = tempo
                
            acc_medias_suj[sujeito-1]        = mean(acuracias[sujeito-1,:]) # em %
            tempos_medios_suj[sujeito-1]     = mean(tempos[sujeito-1,:]) # em ms (milisegundos)
            acc_dvpadrao_suj[sujeito-1]      = std(acuracias[sujeito-1,:]) # em %
            tempos_dvpadrao_suj[sujeito-1]   = std(tempos[sujeito-1,:]) # em ms (milisegundos)
        
        result_nb.append(array([mean(acuracias),std(acuracias),mean(tempos),std(tempos),sum(sum(tempos))]))
        #print('Acc média(%):', round(mean(acuracias)*100,1), '+/-', round(std(acuracias)*100,1))
        #print('Custo médio(ms):', int(mean(tempos)*1000), '+/-', int(std(tempos)*1000))
        #print('Custo total(s):', round(sum(sum(tempos)),1))
        
        # print('Acc média(%):', round(mean(acc_medias_suj)*100,1), '+/-', round(std(acc_medias_suj)*100,1))
        # print('Custo médio(ms):', int(mean(tempos_medios_suj)*1000), '+/-', int(std(tempos_medios_suj)*1000))
    
    res = asarray(result_nb)
    print('Melhor acc média(%):', round(max(res[:,0])*100,1))
    print('Pior acc média(%):', round(min(res[:,0])*100,1))
    print('Acc média/bandas(%):', round(mean(res[:,0])*100,1), '+/-', round(mean(res[:,1])*100,1))
    print('Custo médio/bandas(ms):', int(mean(res[:,2])*1000), '+/-', int(mean(res[:,3])*1000))
    print('Custo Total médio/bandas(s):', round(mean(res[:,4]),1))
    
    