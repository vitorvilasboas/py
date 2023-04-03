# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from scipy import linalg as lg
import decimal as dc

A01T_RH = np.load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01T_1.npy', 'rb'))
A01T_LH = np.load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01T_2.npy', 'rb'))

epoca = np.array(A01T_RH[0,:,125:625])
epocaLH = np.array(A01T_LH[0,:,125:625])

'''
dimensao = epoca.shape
media = np.mean(epoca[0])
mediana = np.median(epoca[0])
quartis = np.quantile(epoca[0], [0, 0.25, 0.5, 0.75, 1])
desvio_padrao = np.std(epoca[0], ddof=1)
variancia = np.var(epoca[0])
variancia_ = np.sum(pow(epoca[0], 2))/len(epoca[0]) # pow(epoca[0], 2) quadrado de todos elementos em epoca[0], np.sum(epoca[0]) somatório de todos elementos em epoca[0]
detalhes = stats.describe(epoca[0])
covar0 = np.cov(epoca[0],epoca[1]) # covariâncoa entre dois canais (exemplo)
pinversa = lg.pinv(epoca) # pseudo-inversa da matriz (método de Moore-Penrose)

S0 = np.dot(epoca, epoca.T) # ou epoca.dot(epoca.T) # produto escalar entre matrizes
S0 /= np.trace(S0)
'''

# Modela/Treina CSP
n_canais, n_amostras = epoca.shape
covar = np.cov(epoca)/n_canais # covariância normalizada por num canais
covar /= np.trace(covar) # np.trace(cov_1) = np.sum(np.diag(cov_1)) = soma da diagonal principal

covarLH = np.cov(epocaLH)/n_canais 
covarLH /= np.trace(covarLH)

[D, W] = lg.eigh(covar, covar + covarLH)

ind = np.empty(n_canais, dtype=int)
ind[::2] = np.arange(n_canais - 1, n_canais // 2 - 1, -1) 
ind[1::2] = np.arange(0, n_canais // 2)

W = W[:, ind]

n_componentes = 6
filtro = W.T[:n_componentes]


# Aplica CSP
A01E = np.load(open('/mnt/dados/bci_tools/dset42a/npy/epocas_t2/A01E_1.npy', 'rb'))
epocaVAL = np.array(A01E[0,:,125:625])

X = np.asarray(np.dot(filtro, epocaVAL))
X = (X ** 2).mean(axis=-1) #compute features (mean band power)

# To standardize features
mean_ = X.mean(axis=0)
std_ = X.std(axis=0)