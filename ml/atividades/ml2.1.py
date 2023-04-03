# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np
import matplotlib.pyplot as plt

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

x_test = np.linspace(0,1,100, endpoint=False)
t_test = np.sin(2*np.pi*x_test) + 0.1 * np.random.randn(len(x_test)) 

M = 9
ln_lbd = [-18,0] # np.exp(-18) == 1e-8

A_train = np.asarray([x_train**i for i in range(M+1)]).T
A_test = np.asarray([x_test**i for i in range(M+1)]).T

plt.subplots(1, figsize=(13,5)) 
## Regularization (quanto maior o expoente, maior a regularizacao quando o expoente tende a 0 a regularização tende a 1 
## atenuando ainda mais o overfitting, porém pode diminuir a capacidade do modelo - underfitting)
for l,k in zip(ln_lbd,range(len(ln_lbd))):
    
    w = np.linalg.solve(np.exp(l) * np.identity(M+1) + A_train.T @ A_train, A_train.T @ t_train) # np.eye(np.size(A_train, 1)) = np.identity(M+1)
    # w = np.linalg.inv(A_train.T.dot(A_train) + np.exp(l) * np.identity(M+1)).dot(A_train.T.dot(t_train)) # option two
    
    y_train = A_train @ w 
    y_test = A_test @ w
    
    plt.subplot(1,2,k+1)
    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="dados treino")
    #plt.scatter(x_train, y_train, facecolor="none", edgecolor="r", s=50, label="predicted train $y(x,w)$")
    plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y_test, c="r", label="$y(x,w)$")
    
    plt.ylabel('$t$', fontsize=13, rotation = 360)
    plt.xlabel('$\mathit{x}$', fontsize=13,labelpad=-15)
    
    plt.xticks([0,1])
    plt.yticks([-1,0,1])
    plt.ylim(-1.5,1.5)
    plt.annotate(('ln $\lambda$ = {}'.format(l)), xy=(0.8, 1))
    
plt.legend(loc='lower left')
plt.savefig('figures/Fig2.1.1_.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
