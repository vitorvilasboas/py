# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:08:27 2019

@author: Rafael Rocha
"""

import numpy as np
from sklearn.metrics import mean_squared_error

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = np.array([[0],[1],[1],[0]]) # Saída K = 1
#t = np.array([[0, 0],[0, 1],[0, 1],[0, 0]]) # Saída K = 2

D = np.size(x,1)
N = np.size(x,0)
M = 2
K = 1

b1 = np.ones([N,1]) # Bias
xb = np.concatenate([b1, x], axis=1) # Entrada com bias

print(np.concatenate([xb,t], axis=1))

w1 = np.random.rand(M,D+1)
w2 = np.random.rand(K,M+1)

# Pesos predefinidos para M = 5 e K = 1
#data = np.load('pesos_xor1.npz')
#w1 = data['w1']
#w2 = data['w2']

# Pesos predefinidos para M = 2 e K = 1
#data = np.load('pesos_xor2.npz')
#w1 = data['w1']
#w2 = data['w2']


a = np.matmul(w1, xb.T)
z = np.tanh(a)

b2 = np.ones([1,N]) # Bias
zb = np.concatenate([b2, z]) # Entrada com bias

y = np.matmul(w2, zb)

print('\n')
print(np.concatenate([t,y.T], axis=1))

# Mean squared error
mse = np.mean(np.square(y-t.T))

print('\n')
print(mse)

#deltak = y-t.T
#
#a_temp = (1-np.square(zb))
#b_temp = np.matmul(w2.T, deltak)
#
##deltaj = np.matmul(a_temp, b_temp)
#deltaj = a_temp*b_temp
#
#dw2 = np.matmul(deltak, zb.T)
#dw1 = np.matmul(deltaj, xb)