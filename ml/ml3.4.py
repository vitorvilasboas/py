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

M = [0,1,3,9]
learn = 0.5
plt.subplots(1, figsize=(10,6))
for j, grau in enumerate(M):
    A_train = np.asarray([x_train**i for i in range(grau+1)]).T
    A_test = np.asarray([x_test**i for i in range(grau+1)]).T
    
    # w = np.linalg.solve(A_train.T @ A_train, A_train.T @ t_train)
    
    w = np.random.rand(len(A_train.T)) #np.random.normal(loc=0, scale=0.1, size=len(A_train.T))
    for i in range(len(A_train)): 
        w = w + learn * A_train[i] * (t_train[i] - w.T @ A_train[i])
    
    print("M={}: w = [{}]".format(grau,w))
    
    y_train = A_train @ w 
    y_test = A_test @ w 
    
    plt.subplot(2,2,j+1)
    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="dados treino")
    plt.plot(x_correct, t_correct, c="g", alpha=0.6, label="$\sin(2\pi x)$")
    plt.plot(x_test, y_test, c="r", label="$y(x,w)$")
    plt.ylabel('$t$', fontsize=13, rotation = 360)
    plt.xlabel('$\mathit{x}$', fontsize=13,labelpad=-15)
    plt.xticks([0,1])
    plt.yticks([-1,0,1])
    plt.ylim(-1.5,1.5)
    plt.annotate(('M={}'.format(grau)), xy=(0.8, 1))
    
plt.legend(bbox_to_anchor=(1.05, 2.2), loc=2, borderaxespad=0.)
plt.savefig('figures/Fig3.4.1_.png', format='png', dpi=300, transparent=True, bbox_inches='tight')