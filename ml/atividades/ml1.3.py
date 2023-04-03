# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np
import matplotlib.pyplot as plt

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

x_test = np.linspace(0,1,100, endpoint=False)
noise = np.random.randn(len(x_test))
t_test = np.sin(2*np.pi*x_test) + 0.1 * noise # np.random.normal(scale=0.1, size=x_train.shape)

M = [0,1,3,9] # complexidade do modelo - n° de parametros livres. Representa M+1 características dos dados de entrada

plt.subplots(1, figsize=(10,6))
for j, grau in enumerate(M):
    # A[M+1]. Bias = x0 = 1 representa o limiar convertido em peso w0. A=[[x⁰],[x¹],[x²],...,[x^M]]
    A_train = np.asarray([x_train**i for i in range(grau+1)]).T
    A_test = np.asarray([x_test**i for i in range(grau+1)]).T
    
    # conjunto de parametros livres que minimizam o erro entre y e t
    w = np.linalg.solve(A_train.T @ A_train, A_train.T @ t_train)  # inv(A.T * A) * A.T * t
    # w = np.linalg.inv(A_train.T.dot(A_train)).dot(A_train.T.dot(t_train)) # test one
    # w = np.linalg.inv(A_train.T @ A_train) @ (A_train.T @ t_train) # test two
    # w = np.linalg.pinv(A_train) @ t_train # test three
    
    print("M={}: w = [{}]".format(grau,w))
    
    y_train = A_train @ w # Aplicando modelo aos próprios dados que o treinaram
    
    y_test = A_test @ w # Generalizando w aos dados de teste
    
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
#plt.savefig('figures/Fig1.3.1_.png', format='png', dpi=300, transparent=True, bbox_inches='tight')