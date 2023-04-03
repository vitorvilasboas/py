# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])
M = [0,1,3,9]
W = np.zeros([M[-1]+1,len(M)]) 
for m,k in zip(M,range(len(M))):
    A_train = np.asarray([x_train**i for i in range(m+1)]).T
    w = np.linalg.solve(A_train.T @ A_train, A_train.T @ t_train)  # inv(A.T * A) * A.T * t
    for j in range(len(w)): W[j,k] = w[j]
print(W)