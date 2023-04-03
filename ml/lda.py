# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np
import matplotlib.pyplot as plt
from prml.linear import FishersLinearDiscriminant
import sklearn.datasets as ds
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

#from scipy.linalg import eigh

np.set_printoptions(precision=4) # determinam como os números de ponto flutuante, matrizes e outros objetos NumPy são exibidos

iris = ds.load_iris()
X = iris.data  # features
t = iris.target
t_names = iris.target_names

n_cls = 3

Xa = np.asarray(X[t == 0,:])
Xb = np.asarray(X[t == 1,:])
Xc = np.asarray(X[t == 2,:])

Xa_train = Xa[:25,:]
Xb_train = Xb[:25,:]
Xc_train = Xc[:25,:]

Xa_test = Xa[25:,:]
Xb_test = Xb[25:,:]
Xc_test = Xc[25:,:]


# Etapa 1: Calcular os vetores médios d-dimensionais (d = núm. características) para as diferentes classes
arrays_medios = []
for cl in range(0,n_cls):
    arrays_medios.append(np.mean(X[t==cl], axis=0))
    print('Vetores Médios de Classe C%s: %s\n' %(cl+1, arrays_medios[cl]))

# Etapa 2a: Calcular a matriz de dispersão dentro da classe S_W    
S_W = np.zeros((4,4))
for cl,mki in zip(range(0,n_cls), arrays_medios):
    S_k = np.zeros((4,4))                  # Matriz de dispersão para cada classe
    for xi in X[t == cl]:
        xi, mki = xi.reshape(4,1), mki.reshape(4,1) # transformando em vetores coluna
        S_k += (xi-mki).dot((xi-mki).T)    # Matriz de dispersão da classe k
    S_W += S_k                             # soma das matrizes de dispersão das classes
print('Matriz de Dispersão dentro da classe:\n', S_W)

# Etapa 2b: Calcular a matriz de dispersão entre classes S_B 
m_geral = np.mean(X, axis=0)
S_B = np.zeros((4,4))
for i,mki in enumerate(arrays_medios):  
    n = X[t==i,:].shape[0]
    mki = mki.reshape(4,1) 
    m_geral = m_geral.reshape(4,1) # make column vector
    S_B += n * (mki - m_geral).dot((mki - m_geral).T)
print('Matriz de Dispersão entre classes:\n', S_B)

# Etapa 3: Calcular os autovetores e os autovalores da matriz S_W^{-1} * S_B para obter os discriminantes lineares
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4,1)   
    print('\nAutovetor {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Autovalor {:}: {:.2e}'.format(i+1, eig_vals[i].real))

# Etaba 3a: Testando se o cálculo dos autovetores e autovalores está correto    
for i in range(len(eig_vals)):
    eigv = eig_vecs[:,i].reshape(4,1)
    np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
                                         eig_vals[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)
print('ok')

# Etapa 4: Selecionat os discriminantes lineares para o novo espaço de características
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])
    
print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
    
# Etapa 4.2 - Choosing k eigenvectors with the largest eigenvalues   
W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('Matrix W:\n', W.real)

# Etapa 5 - Transformando as amostras no novo subespaço
X_lda = X.dot(W)
assert X_lda.shape == (150,2), "The matrix is not 150x2 dimensional."
    
    
    
    
    