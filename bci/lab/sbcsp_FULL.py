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
from sklearn.neural_network import MLPClassifier


def load_data(sujeito, classes):
    ds = ['T_', 'E_']
    X = []  # vetor de dimensão 4 onde as duas primeiras contém os dados de treinamento das duas classes e as duas últimas os dados de validação das duas classes      
    for i in range(2):
        for j in range(2):
            '''
            path = '/mnt/dados/bci_tools/dset42a/fdt/epocas_t2/A0' + str(sujeito) + ds[i] + str(classes[j]) + '.fdt'
            fid = open(path, 'rb')
            dados = fromfile(fid, dtype('f'))
            dados = dados.reshape((72, 1000, 22)) #Dá uma nova forma a uma matriz sem alterar seus dados
            dados = transpose(dados, (0,2,1))
            '''
            path = '/home/vboas/devto/datasets/BCICIV_2a/npy/epocas_t0/A0' + str(sujeito) + ds[i] + str(classes[j]) + '.npy'
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

def filtragemFFT(X, fs, fl, fh):
    m = 80
    XF = []
    for i in range(4):
        ne = X[i].shape[0]
        nc = X[i].shape[1]
        na = X[i].shape[2]
        XF1 = zeros((ne, nc, 2*m))  # FFT simples = FN; FFT concatenada ou INTERCALADA = 2*FN;
        XFFT = fft(X[i])
        for k in range(ne):
            for l in range(nc):
                XREAL = real(XFFT)[k,l,:m]  # :102 representando as 51 primeiras frquências absolutas
                XIMAG = imag(XFFT)[k,l,:m]
                XF1[k, l, :] = list(itertools.chain.from_iterable(zip(XREAL,XIMAG)))
        XF.append(XF1)        
    return XF

def filtragemPROJ(XJ, fs, fl, fh):
    XF = []
    for j in range(4): # filtra os dados de treinamento e validação das duas classes
        XJi = XJ[j]
        ne = XJi.shape[0] # n_epocas
        na = XJi.shape[2] # n_amostras
        nf = fs/2.
        bin0 = int(fl * (fs/nf))  # para fl = 8 bin0 = 15
        binN = int(fh * (fs/nf)) # para fh = 30 bin0 = 59
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
    return XF

def filtragemIIR(XJ, fs, fl, fh):
    nf = fs/2.
    b, a = butter(ordem, [fl/nf, fh/nf], btype='bandpass')
    XF = []
    for i in range(4): # filtra os dados de treinamento e validação das duas classes 
        filtrado = lfilter(b, a, XJ[i])
        # filtrado = filtfilt(b, a, X[i])
        XF.append(filtrado)
    return XF

def get_step(n_bins, n_bands, overlap):
    # Função usada para obter o step (intervalo de coeficientes entre duas bandas)
    # Estimativa inicial do step, que seria o número de bins sobre o número de bandas truncado em múltiplos de 2
    step = int((n_bins / n_bands / 2) * 2)
    
    size = step * overlap

    # Porém as vezes o resultado retornado faz com que a última banda tenha bins não existentes. Para corrigir isso usei um loop que verifica o último bin da 
    # última banda e se ele não for menor ou igual ao número de bins o step é reduzido por 2.
    while True:
        last_end = (n_bands-1) * step + size
        if last_end <= n_bins:
            break
        step -= 2
        #print(step)
        size = step * overlap  # add by Vitor, cleison.py based
    return step

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
    atraso, t_filtro, n_componentes, fs, fl, fh, ordem, n_bandas, Clog = args  
    
    X = load_data(sujeito, classes)
    
    t0 = time()
    
    XJ = windowing(X, fs, 0, 2.5, 4.5, atraso) # se load /epocas_t0/, Tini = 0; se load /epocas_t2/, Tini = 2
    
    if t_filtro == 'IIR': XF = filtragemIIR(XJ, fs, 0.001, fh)
    elif t_filtro == 'FFT': XF = filtragemFFT2(XJ, fs, fl, fh)
    elif t_filtro == 'PROJ': XF = filtragemPROJ(XJ, fs, fl, fh)
    
    #print (asarray(XF).shape)
    #for i in XF[0][0][0][480::]: print(i)
    
    XT = concatenate([XF[0],XF[1]]) # treinamento A e B
    XV = concatenate([XF[2],XF[3]]) # validação A e B
    
    # Divide sub-bands
    n_bins = len(list(XT[0][0]))
    overlap = 2
    step = get_step(n_bins, n_bandas, overlap)
    size = int(step * overlap)  # tamanho fixo p/ todas sub bandas. overlap em 50%

    XT = [XT[:, :, i*step:i*step+size] for i in range(n_bandas)] # [ 33 x 144 x 22 x 8 ]
    XV = [XV[:, :, i*step:i*step+size] for i in range(n_bandas)] # [ 33 x 144 x 22 x 8 ]
    
    y = concatenate([zeros(72), ones(72)]) #vetor gabarito
    
    # CSP
    csp = [CSP(n_components=n_componentes) for i in range(n_bandas)]

    for i in range(n_bandas): csp[i].fit(XT[i], y)
    
    #print (asarray(XT).shape)
    
    XT_CSP = [csp[i].transform(XT[i]) for i in range(n_bandas)]
    XV_CSP = [csp[i].transform(XV[i]) for i in range(n_bandas)]
    
    #print (asarray(XT_CSP).shape)
    #print(XT_CSP[0][0])
    
    # LDA
    SCORE_T = zeros((144, n_bandas))
    SCORE_V = zeros((144, n_bandas))
    clf = [LinearDiscriminantAnalysis() for i in range(n_bandas)]
    for i in range(n_bandas):
        clf[i].fit(XT_CSP[i], y)
        SCORE_T[:, i] = ravel(clf[i].transform(XT_CSP[i])) # classificaçoes de cada época nas N sub bandas - auto validação
        SCORE_V[:, i] = ravel(clf[i].transform(XV_CSP[i])) # validação
        
    # print (asarray(SCORE_T).shape)   
    
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
    ans = svc.predict(META_SCORE_V)
    
    acuracia = mean(ans == y)
    
    
    '''
    redeNeural = MLPClassifier(verbose=False, 
                               max_iter=10000, 
                               tol=0.00001, 
                               activation='logistic',
                               learning_rate_init=0.0001)
    
    redeNeural.fit(META_SCORE_T, y)
    saida_rna = redeNeural.predict(META_SCORE_V)
    acuracia2 = mean(saida_rna == y)
    '''
    
    tempo = time() - t0
    
    #print(sujeito, classes, round(tempo, 2), round(acuracia * 100, 2))
    #print(sujeito, classes, round(tempo, 2), round(acuracia2 * 100, 2))
    
    return acuracia, tempo
    
if __name__ == "__main__":
    # sujeito = arange(1, 10)
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    atrasos = range(0,5)
    
    t_filtro = ['IIR','FFT','PROJ']  # opções: 'IIR' 'FFT' 'PROJ'
    n_componentes = 4
    fs = 250
    fl = 0 
    fh = 40
    ordem = 5
    n_bandas = 33
    Clog = -4
    
    
    print('\n>> Qtd sujeitos: 9')
    print('>> Classes: LH x RH')
    print('>> Abordagem: SBCSP+LDA+Bayes+SVM   ')
    print('>> Parâmetros: [',str(fl),'-',str(fh),'Hz]/',str(n_bandas),'  Overlap=50% ')
    
    for tf in t_filtro:
        print('\n========================',tf,'=========================')
        print('Atrasos\t','| Acc Média (%)', '| Custo (s)   ', '| Custo_i (s)')
        print('======================================================')
        for atraso in atrasos:
            args = atraso, tf, n_componentes, fs, fl, fh, ordem, n_bandas, Clog
            result = [sbcsp(suj, classes[i], args) for suj in range(1, 10) for i in range(0, 6) ]
            print ('  '+ str(atraso) +'\t |  ' + 
                   str(round(mean(asarray(result)[:,0])*100, 2)) + '%\t |  ' + 
                   str(round(sum(asarray(result)[:,1]), 2)) + 's\t| ' +
                   str(round(mean(asarray(result)[:,1]), 2)) + 's')
    
   
    
    #args = 0, 'FFT', n_componentes, fs, fl, fh, ordem, n_bandas, Clog
    #result = [sbcsp(suj, classes[i], args) for suj in range(1, 10) for i in range(0, 6) ]         
    #print ('Acc média:\t' + str(round(mean(asarray(result)[:,0])*100, 2)) + '%')
    #print ('Custo total:\t' + str(round(sum(asarray(result)[:,1]), 2)) + 's')
    #print ('Custo médio:\t' + str(round(mean(asarray(result)[:,1]), 2)) + 's\n')