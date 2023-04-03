# dados dois vetores x,t

import math 
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


# teste 123 github

M = [0,1,3,9]

#M = 10

Erms = np.zeros([len(M)])
EVrms = np.zeros([len(M)])

x = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881])
t = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839])

xv = np.sort(np.random.rand(100))
r = np.random.randn(len(xv))
tv = np.sin(2*np.pi*xv) + 0.1*r

cores = ['c','m','orange','firebrick','olive','dimgray','darkorange','lime','k','peru','purple']

t = t.T

plt.figure(1)
plt.subplot(1,1,1)
plt.scatter(x, t, color='b', marker = 'o') #facecolors = 'b'

for m,c in zip(M,range(4)):
    A = np.zeros([len(x),m+1])
    A[:,0] = np.ones([len(x)])
    
    for i in range(m):
        xa = [xi**(i+1) for xi in x]
        A[:,i+1] = np.asarray(xa).T
        
    w = np.dot(inv(np.dot(A.T, A)), np.dot(A.T,t))
    
    h = np.dot(A,w) # h é a saida do modelo
    plt.xlabel('t', fontsize=14)
    plt.ylabel('x', size=14)
    plt.plot(x,h, color=cores[c], label=('M={}'.format(m)))
    plt.legend(loc='upper right', ncol = 2, fontsize=10)

    E = (np.dot((np.dot(A,w)-t).T,(np.dot(A,w)-t)))/2
    
    Erms[c] = math.sqrt((2*E)/len(x))
    
    A = np.zeros([len(xv),m+1])
    A[:,0] = np.ones([len(xv)])
    
    for i in range(m):
        xa = [xi**(i+1) for xi in x]
        A[:,i+1] = np.asarray(xa).T
    
    E = (np.dot((np.dot(A,w)-tv).T,(np.dot(A,w)-tv)))/2
    
    EVrms[c] = math.sqrt((2*E)/len(xv))


plt.figure(2)
plt.subplot(2,1,1)
plt.yticks(np.arange(0, 1, step=0.2))
plt.scatter(M,Erms, color='b', facecolors = 'b', marker = 'o')
plt.plot(M,Erms)
plt.scatter(M,EVrms, color='b', facecolors = 'b', marker = 'o')
plt.plot(M,EVrms)


# norma de w^2 = w.T*w = w0^2 + w1^2 + ... + wM^2    

# regularização controla a magnitude e o overfitting

# qto maior o lambda menor a esfera de contenção de w. lambda sempre positivo e em geral entre 0 e 1. ln lambda = ⁻18 -> e^-18

# M, lambda são hyperparametros determindaos antes de w - determinados por validação cruzada nos dados
      

#plt.scatter(x, h, color='r', facecolors = 'r', marker = 'o', label=('h'))
#plt.plot(x,E, color='orange')# -*- coding: utf-8 -*-

