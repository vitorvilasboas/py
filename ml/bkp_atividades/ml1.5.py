# -*- coding: utf-8 -*-
import numpy as np

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

x_test = np.linspace(0.01,0.99,1000, endpoint=False)
noise = np.random.randn(len(x_test))
# noise = np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
t_test = np.sin(2*np.pi*x_test) + 0.1 * noise 

M = [0,1,3,9]
W = np.zeros([M[-1]+1,len(M)]) 
for m,k in zip(M,range(len(M))):
    A_train = np.asarray([x_train**i for i in range(m+1)]).T
    w = np.linalg.pinv(A_train) @ t_train # inv(A.T * A) * A.T * t 
    for j in range(len(w)): W[j,k] = w[j]