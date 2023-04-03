# -*- coding: utf-8 -*-

# regularização controla a magnitude e o overfitting
# qto maior o lambda menor a esfera de contenção de w. lambda sempre positivo e em geral entre 0 e 1. ln lambda = ⁻18 -> e^-18

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

A = np.zeros([len(x),M+1])
A[:,0] = np.ones([len(x)])
for i in range(M):
    #xa = [xi**(i+1) for xi in x]
    A[:,i+1] = (x**(i+1)).T
    
w = inv(A.T.dot(A)).dot(A.T.dot(t))

y = A.dot(w) # saída do modelo

plt.figure(1)
plt.subplot(1,1,1)
plt.scatter(x, t, color='b', facecolors='none', marker = 'o')
plt.plot(x,y, color='r')

# teste do modelo
#xv = np.linspace(0,1,100)
#xv = np.sort(np.random.rand(1,100))
xv = np.arange(0,1,0.01)
Av = np.zeros([len(xv),M+1])
for i in range(M):
    Av[:,i+1] = (xv**(i+1)).T
yv = Av.dot(w)
#plt.plot(xv,yv, color='r')

E = (np.dot((np.dot(A,w)-t).T,(np.dot(A,w)-t)))/2

Erms = math.sqrt((2*E)/len(x))

plt.figure(2)
plt.subplot(2,1,1)
plt.yticks(np.arange(0, 1, step=0.2))
plt.scatter(M,Erms, color='b', facecolors = 'b', marker = 'o')
plt.plot(M,Erms)