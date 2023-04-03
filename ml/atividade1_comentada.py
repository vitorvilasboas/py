# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# F(x): Função alvo - Para fins didaticos é conhecida, mas normalmente é desconhecida! ´E o que deseja-se aprender.
x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)


## Dataset Treinamento com rótulos=alvo=target -> mapeamento de x para y (x --> y)
x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

plt.figure()
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()


## Dataset Teste (N maior - mais dados)
x_test = np.linspace(0,1,1000, endpoint=False)
noise = np.random.randn(len(x_test))
# noise = np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
t_test = np.sin(2*np.pi*x_test) + 0.1 * noise # esse mesmo processo gerou a amostra de treinamento - aki, somente para ilustração - em geral não se conhece os alvos em dados de teste

plt.figure()
plt.scatter(x_test, t_test, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()


## Modelo de aprendizagem - Função Hipótese (g(x)) que quer aproximar f(x)
M = 3# complexidade do modelo - n° de parametros livres - que na forma polinomial define o grau progressivo do polinomio
# ... para representar as M+1 características dos dados de entrada
A_train = np.asarray([x_train**i for i in range(M+1)]).T # modelo polinomial - não linear - a transformação de x provocada pela potência progressiva regida por M torna o modelo não linear em x
# A é M+1 para considerar já com o bias = x0 = 1 representando o limiar que foi convertido para peso w0
w = np.linalg.inv(((A_train.T).dot(A_train))).dot((A_train.T).dot(t_train)) # encontrar vetor de pesos para cada característica de X que comporão a função hipótese polinomial g(x) - w é linear - diferentemente em w onde não há transformação polinomial


## Aplicando modelo aos próprios dados que o treinaram
y_train = A_train.dot(w)

plt.figure()
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
plt.scatter(x_train, y_train, facecolor="none", edgecolor="r", s=50, label="predicted test")
plt.plot(x_train, y_train, c="r", label="predicted test $y(x,w)$")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()


## Aplicando modelo à novos dados, deconhecidos (teste) "generalização" - objetivo da aprendizagem
A_test = np.asarray([x_test**i for i in range(M+1)]).T

y_test = A_test.dot(w)

plt.figure(figsize=(10,6))
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
plt.plot(x_test, y_test, c="r", label="$y(x,w)$")
plt.plot(x_correct, t_correct, c="g", alpha=0.6, label="$\sin(2\pi x)$")
plt.legend()
plt.ylim(-2,2)


## Cálculo do erro de previsão em relação ao alvo - Dados de treinamento - Busca-se minimizar esse erro
E_train = ((y_train - t_train).T).dot((y_train - t_train)) # soma dos quadrados dos erros

Erms_train = sqrt(E_train/len(x_train)) # erro médio quadrático - é normalizado pelo tamanho da amostra N=len(x) (para comparar erro em amostras de vários tamanhos)
# ... a raiz quadrada garante que o ERMS seja medido na mesma escala (e nas mesmas unidades) que a variável de destino t. 


## Cálculo do erro de previsão em relação ao alvo - Dados de teste
E_test = ((y_test - t_test).T).dot((y_test - t_test))

Erms_test = sqrt(E_test/len(x_test))


plt.figure()
plt.scatter(x_test, t_test, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="r", label="training data")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()



## Calculando o erro a partir de modelos com várias complexidades M=[0 até 9]
M = range(0,10)
Et = []
Ev = []
for m in M:
    A = np.asarray([x_train**i for i in range(m+1)]).T
    w = np.linalg.inv(((A.T).dot(A))).dot((A.T).dot(t_train))
    y = A.dot(w)
    Et.append(sqrt((((y - t_train).T).dot(y - t_train))/len(x_train)))
    
    A = np.asarray([x_test**i for i in range(m+1)]).T
    y = A.dot(w)
    Ev.append(sqrt((((y - t_test).T).dot(y - t_test))/len(x_test)))


plt.figure(figsize=(10,7))
plt.plot(M, Et, c="b", marker='o', label='Training')
plt.plot(M, Ev, c="r", marker='o', label='Test')
plt.legend(loc="best")
plt.yticks(np.linspace(0,2,10, endpoint=False))
plt.xlabel('M')
plt.ylabel('$E_{RMS}$')
plt.ylim(0,2)



M = [0,1,3,9]
W = np.zeros([M[-1]+1,len(M)]) 
for M,k in zip(M,range(len(M))):
    A = np.asarray([x_train**i for i in range(M+1)]).T # gerando a matriz A = [ [x⁰],[x¹],[x²],...,[x^M] ].T
    w = np.dot( np.linalg.inv( np.dot(A.T, A) ), np.dot(A.T,t_train))
    for j in range(len(w)): W[j,k] = w[j]





