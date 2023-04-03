# -*- coding: utf-8 -*-
import numpy as np

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

x_test = np.linspace(0.01,0.99,1000, endpoint=False)
noise = np.random.randn(len(x_test))
t_test = np.sin(2*np.pi*x_test) + 0.1 * noise 

M = 9

A_train = np.asarray([x_train**i for i in range(M+1)]).T 

lambd = [-np.inf,-18,0]

W = np.zeros([M+1,len(lambd)]) 

for l,k in zip(lambd,range(len(lambd))):
    
    w = np.linalg.solve(np.exp(l) * np.eye(np.size(A_train, 1)) + A_train.T @ A_train, A_train.T @ t_train)
    # w = np.linalg.inv( (A_train.T).dot(A_train) + (np.exp(l) * np.identity(M+1)) ).dot( (A_train.T).dot(t_train) ) # old
     
    for j in range(len(w)): W[j,k] = w[j]
