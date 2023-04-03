# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# F(x): Função alvo - Para fins didaticos é conhecida, mas normalmente é desconhecida! é o que deseja-se aprender.
x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

## Dataset Treinamento com rótulos=alvo=target -> mapeamento de x para y (x --> y)
x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

## Dataset Teste (N maior - mais dados)
x_test = np.linspace(0.1,0.9,100, endpoint=False)
noise = np.random.randn(len(x_test)) # np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
t_test = np.sin(2*np.pi*x_test) + 0.1 * noise # esse mesmo processo gerou a amostra de treinamento - aqui, somente para ilustração - em geral não se conhece os alvos em dados de teste

## Cálculo do erro de previsão em relação ao alvo - Dados de treinamento - Busca-se minimizar esse erro
## Calculando o erro a partir de modelos com várias complexidades M=[0 até 9]
M = range(0,10)
Erms = []
ErmsV = []
for m in M:
    A = np.asarray([x_train**i for i in range(m+1)]).T
    w = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(t_train))
    y = A.dot(w)
    
    # erro médio quadrático - é normalizado pelo tamanho da amostra N=len(x) (para comparar erro em amostras de vários tamanhos)
    #Erms.append(sqrt((y - t_train).T.dot(y - t_train)/len(x_train)))
    
    E = ((y-t_train).T).dot(y-t_train) # (np.dot((np.dot(A,w)-t).T,(np.dot(A,w)-t)))/2
    Erms.append(sqrt(E/len(x_train)))
    
    A = np.asarray([x_test**i for i in range(m+1)]).T
    y = A.dot(w)
    ErmsV.append(sqrt((y - t_test).T.dot(y - t_test)/len(x_test)))


plt.figure(figsize=(10,7))
plt.plot(Erms, c="b", label='Training', marker='o', markerfacecolor=None, markeredgecolor='b')
plt.plot(ErmsV, c="r", label='Test', marker='o', markerfacecolor=None, markeredgecolor='r')
plt.legend(loc="best")
plt.yticks(np.linspace(0,1,6))
plt.xticks(np.linspace(0,9,10))
plt.xlabel('M', fontsize=16)
plt.ylabel('$E_{RMS}$', fontsize=16)
plt.ylim(0,1)
plt.savefig('figures/Fig1.4.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
