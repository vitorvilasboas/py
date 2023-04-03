# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np
import matplotlib.pyplot as plt

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

x_test = np.linspace(0.1,0.9,1000, endpoint=False)
t_test = np.sin(2*np.pi*x_test) + 0.1 * np.random.randn(len(x_test)) 

M = 9

A_train = np.asarray([x_train**i for i in range(M+1)]).T
A_test = np.asarray([x_test**i for i in range(M+1)]).T

lambd = np.arange(-40,-3)
Erms = []
ErmsV = []
for l in lambd:
    # quanto maior np.exp(l) maior o impacto sobre o erro
    w = np.linalg.solve(np.exp(l) * np.identity(M+1) + A_train.T @ A_train, A_train.T @ t_train) # np.eye(np.size(A_train, 1)) = np.identity(M+1)
    # w = np.linalg.inv(A_train.T.dot(A_train) + np.exp(l) * np.identity(M+1)).dot(A_train.T.dot(t_train)) # option two
    
    y_train = A_train @ w 
    y_test = A_test @ w

    Erms.append(np.sqrt(np.mean(np.square(y_train - t_train)))) #~ np.sqrt((y_train-t_train).T.dot(y_train-t_train)/len(x_train))
    ErmsV.append(np.sqrt(np.mean(np.square(y_test - t_test)))) #~ np.sqrt((y_test-t_test).T.dot(y_test-t_test)/len(x_test))
   
#    E = ((y_train-t_train).T).dot(y_train-t_train) + ((np.exp(l)/2)*(w.T).dot(w))
#    Erms.append(np.sqrt(E/len(x_train)))
    
#    Ev = ((y_test-t_test).T).dot(y_test-t_test) + ((np.exp(l)/2)*(w.T).dot(w))
#    ErmsV.append(np.sqrt(Ev/len(x_test)))
    
plt.figure(figsize=(8,5))
plt.plot(lambd,Erms,c='b', label='Treino')
plt.plot(lambd,ErmsV,c='r', label='Teste')
plt.ylabel('$E_{RMS}$', fontsize=13, rotation = 360, labelpad=10)
plt.xlabel('ln $\lambda$', fontsize=13,labelpad=0)
plt.legend(loc="best")
plt.yticks(np.linspace(0,2,5, endpoint=True))
plt.xticks([-35,-30,-25,-20])
plt.ylim(0,2)
plt.savefig('figures/Fig2.3.1b.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
