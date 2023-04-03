# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:42:57 2019

@author: Rafael Rocha
"""

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([[0],[1],[1],[0]])

# Entrada
i = 3
x = X[i]
t = T[i]

print('Entrada: '+ str(x))
print('Saída desejada: '+ str(t))

# Número de atrabutos
M = np.size(x)

# Criação dos pesos
w = np.random.rand(M)

u = 0
for j in range(M):
    u = u + w[j]*x[j]
    
v = u + 1

# Saída do neurônio
y = np.tanh(v)
print('Saída obtida: '+ str(y))

# Erro quadrático
erro = np.square(y - t)
print('Erro quadrático: '+ str(erro))