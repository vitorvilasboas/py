# -*- coding: utf-8 -*-
from numpy import load, transpose, dtype, fromfile, asarray, concatenate, zeros, ones
from numpy import unique, dot, empty, arange, mean, log
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import lfilter, butter
from time import time


classes = [1, 2]
sujeito = 1
ds = ['T_', 'E_']
X = []  # vetor de dimensão 4 onde as duas primeiras contém os dados de treinamento das duas classes e as duas últimas os dados de validação das duas classes      
for i in range(2):
    for j in range(2):
        path = '/mnt/dados/bci_tools/datasets/bci_cIV_2a/npy/epocas_t2/A0' + str(sujeito) + ds[i] + str(classes[j]) + '.npy'
        dados = load(open(path, 'rb'))
        X.append(dados)

## Janelamento
t0 = time()
fs = 250
W_Start = int(0.5*fs)
W_End = int(2.5*fs)
XJ = []
for i in range(4):
    janela = transpose(X[i][:,:,W_Start:W_End], (1, 0, 2))    
    XJ.append(transpose(janela, (1, 0, 2)))

## Filtragem
fl = 8 
fh = 30
ordem = 5
nf = fs/2.
b, a = butter(ordem, [fl/nf, fh/nf], btype='bandpass')
XF = []
for i in range(4): # filtra os dados de treinamento e validação das duas classes 
    filtrado = lfilter(b, a, XJ[i])
    XF.append(filtrado)

XT = concatenate([XF[0],XF[1]]) # treinamento A e B
XV = concatenate([XF[2],XF[3]]) # validação A e B

y = concatenate([zeros(72), ones(72)]) #vetor gabarito

## Filtragem espacial e EXTRAÇÃO DE CARACTERÍSTICAS (CSP)
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

n_componentes = 6
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

print(round(acuracia * 100, 2))



