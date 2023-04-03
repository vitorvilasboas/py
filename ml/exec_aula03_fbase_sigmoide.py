# -*- coding: utf-8 -*-
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

#plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (11,7)

x = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881])
t = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839])

M = 3
s = 2

#mu = np.random.randint(len(x), size=M)
pos = np.random.randint(len(x)-1, size=M)
mu = np.sort([x[i] for i in pos]) # 1) gerar números aleatórios dentro da faixa de valores de x
#mu = np.array([np.mean(x[0:3]), np.mean(x[3:6]), np.mean(x[6:])]) # 2) separar x em 3 e obter a média de x para cada região..

A = np.zeros([len(x),M+1])
A[:,0] = np.ones([len(x)])
for i in range(M):   
    xa = [ 1/(1 + np.exp((xi - mu[i])/s)) for xi in x]
    A[:,i+1] = np.asarray(xa).T
    #plt.plot(xa, color='r')

w = inv(A.T.dot(A)).dot(A.T.dot(t))

y = A.dot(w) # aplicando o modelo como mapeamento de x para y

plt.figure(1)
plt.subplot(1,1,1)
plt.scatter(x, t, color='b', facecolors='none', marker = 'o')
plt.plot(x,y, color='r')
plt.ylim(-2,2)
plt.xlim(0,1)       












# -*- coding: utf-8 -*-

