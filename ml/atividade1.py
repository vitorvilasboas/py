# -*- coding: utf-8 -*-
# =============================================================================
#  Considere o conjunto de dados de treinamento 
#  x = [0.1387 0.2691 0.3077 0.3625 0.4756 0.5039 0.5607 0.6468 0.7490 0.7881 ] e
#  t = [0.8260 1.0469 0.7904 0.6638 0.1731 -0.0592 -0.2433 -0.6630 -1.0581 -0.8839 ]. 
#  Os dados foram gerados satisfazendo a equação: t = sin(2πx) + ruído. Faça:
#  1. Gere a Figura 1.2 para os dados de treinamento;
#  2. Considere o modelo polinomial dado pela Eq 1.1. Ele é linear em w e não linear em x. Explique.
#  3. Considere as seguintes ordens para o modelo polinomial Eq. 1.1, M = {0, 1, 3, 9}. 
#     Determine os parâmetros livres, o vetor w, que minimizam a função erro, dada pela Equação 1.2.
#     Apresente os gráficos equivalentes aos da Figura 1.4.
#  4. Gere a Figura 1.5 para a fun¸c˜ao erro RMS, ERMS (w). Comente os resultados;
#  5. Gere a Tabela 1.1 para os valores de M = {0, 1, 3, 9}. Realize uma análise crítica dos resultados, 
#     com ênfase para a complexidade do modelo e o ajuste excessivo do modelo aos dados.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

## Q1.
x0 = np.linspace(0,1,1000)
y0 = np.sin(2*np.pi*x0) 
plt.plot(x0,y0, c='g', label="t = sen(r'$2\pix$)")

x = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])
plt.scatter(x,t, c='b')

## Q3.
params = [0,1,3,9] # range(0,10)
for M in params:
    A = np.asarray([x**i for i in range(M+1)]).T # gerando a matriz A = [ [x⁰],[x¹],[x²],...,[x^M] ].T
    Apsi = np.linalg.inv( (A.T.dot(A)) ).dot(A.T)# calculo da pseudo inversa da matriz A = inversa(A.T * A) * A.T
    w = np.dot(Apsi,t) # conjunto de parmetros livres que minimizam o erro entre y e t
    y = np.dot(A,w) # y é a saida do modelo = f(x,w) = A*w
    # plt.plot(x,y,c='k')
    
    xv = np.linspace(0,1,1000)
    Av = np.asarray([xv**i for i in range(M+1)]).T
    yv = np.dot(Av,w)
    plt.plot(xv,yv,c='r')
    plt.ylim(-2,2)

# Q4.
params = range(0,10)
Erms = []
ErmsV = []
xv = np.linspace(0.1,0.8,100, endpoint=False)
tv = np.sin(2*np.pi*xv) + 0.1*np.random.randn(len(xv))
for M in params:
    A = np.asarray([x**i for i in range(M+1)]).T # gerando a matriz A = [ [x⁰],[x¹],[x²],...,[x^M] ].T
    w = np.dot( np.linalg.inv( np.dot(A.T, A) ), np.dot(A.T,t))
    y = np.dot(A,w)
    E = ((y-t).T).dot(y-t) # (np.dot((np.dot(A,w)-t).T,(np.dot(A,w)-t)))/2
    Erms.append(sqrt((E)/len(x)))

    Av = np.asarray([xv**j for j in range(M+1)]).T
    yv = np.dot(Av,w)
    Ev = ((yv-tv).T).dot(yv-tv)
    ErmsV.append(sqrt(Ev/len(xv)))
    
plt.figure(figsize=(10,7))
plt.plot(params,Erms,c='b', marker='o', label='Training')
plt.plot(params,ErmsV,c='r', marker='o', label='Test')
plt.legend(loc="best")
plt.yticks(np.linspace(0,2,10, endpoint=False))
plt.ylim(0,2)

# Q5.
params = [0,1,3,9]
W = np.zeros([params[-1]+1,len(params)]) 
for M,k in zip(params,range(len(params))):
    A = np.asarray([x**i for i in range(M+1)]).T # gerando a matriz A = [ [x⁰],[x¹],[x²],...,[x^M] ].T
    w = np.dot( np.linalg.inv( np.dot(A.T, A) ), np.dot(A.T,t))
    for j in range(len(w)): W[j,k] = w[j]


