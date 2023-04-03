# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg as lg
from scipy.signal import lfilter, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

######### CALIBRAÇÃO #########

### load
LHt = np.load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01T_1.npy', 'rb'))
RHt = np.load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01T_2.npy', 'rb'))

ti = 0.5
tf = 2.5
coefi = 0.5


#for tf in range(ti+2, 5):
ini = int(ti * 250)
fim = int(tf * 250)

LHt = LHt[:,:,ini:fim]
RHt = RHt[:,:,ini:fim]

### filtragem
nf = 250/2.
b, a = butter(5, [8/nf, 30/nf], btype='bandpass')
LHtF = lfilter(b, a, LHt)
RHtF = lfilter(b, a, RHt)


### Modela/Treina CSP
n_epocas, n_canais, n_amostras = RHtF.shape
covLH = np.zeros((n_canais, n_canais))
covRH = np.zeros((n_canais, n_canais))

for epoca in range(int(n_epocas)):
    #covLH += np.dot(LHtF[epoca], LHtF[epoca].T)
    #covRH += np.dot(RHtF[epoca], RHtF[epoca].T)
    covLH += (np.cov(LHtF[epoca])/n_canais) # covariância normalizada por num canais
    covRH += (np.cov(RHtF[epoca])/n_canais) # covariância normalizada por num canais
covLH /= np.trace(covLH) # np.trace(cov_1) = np.sum(np.diag(cov_1)) = soma da diagonal principal
covRH /= np.trace(covRH) # np.trace(cov_1) = np.sum(np.diag(cov_1)) = soma da diagonal principal

[D, W] = lg.eigh(covLH, covLH + covRH)

ind = np.empty(n_canais, dtype=int)
ind[::2] = np.arange(n_canais - 1, n_canais // 2 - 1, -1) 
ind[1::2] = np.arange(0, n_canais // 2)

W = W[:, ind]

n_componentes = 6
filtro_csp = W.T[:n_componentes]


Xt = np.concatenate([LHtF,RHtF])
XtCSP = np.asarray([np.dot(filtro_csp, epoca) for epoca in Xt])
XtCSP = np.log(np.mean(XtCSP ** 2, axis=2))


### Treina classificador LDA
clf = LinearDiscriminantAnalysis()
y = np.concatenate([np.zeros(72), np.ones(72)]) #vetor gabarito
clf.fit(XtCSP, y)



######### OPERAÇÃO #########

LHv = np.load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01E_1.npy', 'rb'))
RHv = np.load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01E_2.npy', 'rb'))

### janelamento
#ini = int(0.5 * 250)
#fim = int(2.5 * 250)
LHv = LHv[:,:,ini:fim]
RHv = RHv[:,:,ini:fim]

### filtragem
nf = 250/2.
b, a = butter(5, [8/nf, 30/nf], btype='bandpass')
LHvF = lfilter(b, a, LHv)
RHvF = lfilter(b, a, RHv)

### Aplica CSP
Xv = np.concatenate([LHvF,RHvF])
XvCSP = np.asarray([np.dot(filtro_csp, epoca) for epoca in Xv])
XvCSP = np.log(np.mean(XvCSP ** 2, axis=2))

saida_lda = clf.predict(XvCSP)

acuracia = np.mean(saida_lda == y)

print('Acc {}-{}: {}%' .format(ti,tf, round(acuracia * 100, 2)))