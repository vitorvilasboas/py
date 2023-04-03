# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:45:52 2019

@author: Rafael Rocha
"""

import numpy as np
from keras.utils import to_categorical

# Quantidade de amostras
m = 2

# Quantidade de saídas ou classes
n = 3

# Amplitude mínima e máxima da saída
n_min, n_max = 1, 20

# Geração do rótulo da saída desjada
r = np.random.randint(1, n, size=(m))
r_cat = to_categorical(r)

print('Saída desejada: ' + str(r_cat))

# Saída da rede
y = np.random.randint(n_min, n_max+1, size=(m, r_cat.shape[1]))
print('Saída obtida: ' + str(y))

# Softmax

sy = np.zeros((m, r_cat.shape[1]))
for i in range(m):
    for j in range(r_cat.shape[1]):
        sy[i,j] = np.exp(y[i,j])/np.sum(np.exp(y[i,:]))

print('Saída softmax: ' + str(sy))

# Entropia cruzada
C = -np.sum(r_cat*np.log(sy), axis=1) # axis = 1: soma por linha
# Função de perda
L = np.mean(C)
print('Perda da entropia cruzada: ' + str(L))