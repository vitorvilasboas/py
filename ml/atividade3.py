# -*- coding: utf-8 -*-
# ========= Linear Models for Regression ======================================
# 1. Refaça a Atividade 01 para o caso em que o modelo dado na Eq. 3.3 apresente função de base do tipo Gaussiana, Eq. 3.4;
# 2. Refaça a Atividade 01 para o caso em que o modelo dado na Eq. 3.3 apresente função de base do tipo Sigmoide, Eqs. 3.5 e 3.6;
# 3. Para o problema de ajuste de curva, Seção 3.1.4, discuta os tipos de regularização;
# 4. Considere a Seção 3.1.3, utilize as Eqs. 3.22 e 3.23 para obter o vetor w da Atividade 1, item 3;
# 5. Gere as Figuras 3.5. Discuta os resultados com base em ”bias-variance decomposition”.
# =============================================================================

# essa configuração é conhecida como uma rede neural de base radial - função de base gaussiana

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

plt.rcParams['figure.figsize'] = (11,7)
x = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881])
t = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839])
M = 3
s = 2 # tem interpretação de variancia (pode ser calculada a partir dos blocos de x que calculam cada mu)

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)


# Q1.
#plt.style.use('ggplot')
#mu = np.random.randint(len(x), size=M)
pos = np.random.randint(len(x)-1, size=M)
mu = np.sort([x[i] for i in pos]) # 1) gerar números aleatórios dentro da faixa de valores de x
#mu = np.array([np.mean(x[0:3]), np.mean(x[3:6]), np.mean(x[6:])]) # 2) separar x em 3 e obter a média de x para cada região..
# outra opção é adotar a qtd de mu igual à N... assim haverá uma gaussiana vcentrada em cada elemento de x
# para determinar os centros pode-se utilizar o algoritmo o k-means
A = np.zeros([len(x),M+1])
A[:,0] = np.ones([len(x)])
for i in range(M):
    xa =  [(np.exp((-1)*(((xi - mu[i])**2)/2*(s**2)))).T for xi in x]
    A[:,i+1] = np.asarray(xa).T
    #plt.plot(xa, color='r')
w = inv(A.T.dot(A)).dot(A.T.dot(t))
y = A.dot(w) # aplicando o modelo como mapeamento de x para y

#plt.ylim(-2,2)
#plt.xlim(0,1)

plt.figure()
plt.scatter(x, t, facecolor="none", edgecolor="b", s=50, label="target train")
plt.scatter(x, y, facecolor="none", edgecolor="r", s=50, label="predicted train")
plt.plot(x, y, c="r", label="predicted test $y(x,w)$")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()


x_test = np.linspace(0,1,1000, endpoint=False)
noise = np.random.randn(len(x_test))
# noise = np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
t_test = np.sin(2*np.pi*x_test) + 0.1 * noise # esse mesmo processo gerou a amostra de treinamento - aki, somente para ilustração - em geral não se conhece os alvos em dados de teste

A_test = np.asarray([x_test**i for i in range(M+1)]).T

y_test = A_test.dot(w)

plt.figure(figsize=(10,6))
plt.scatter(x, t, facecolor="none", edgecolor="b", s=50, label="target train")
plt.plot(x_test, y_test, c="r", label="$y(x,w)$")
plt.plot(x_correct, t_correct, c="g", alpha=0.6, label="$\sin(2\pi x)$")
plt.legend()
plt.ylim(-6,6)



#Q2.
s = 2
#mu = np.random.randint(len(x), size=M)
pos = np.random.randint(len(x)-1, size=M)
mu = np.sort([x[i] for i in pos]) # 1) gerar números aleatórios dentro da faixa de valores de x
#mu = np.array([np.mean(x[0:3]), np.mean(x[3:6]), np.mean(x[6:])]) # 2) separar x em 3 e obter a média de x para cada região..
A = np.zeros([len(x),M+1])
A[:,0] = np.ones([len(x)])
for i in range(M):   
    xa = [ 1/(1 + np.exp((xi - mu[i])/s)) for xi in x]
    A[:,i+1] = np.asarray(xa).T
    #plt.plot(xa, color='r')

w = inv(A.T.dot(A)).dot(A.T.dot(t))

y = A.dot(w) # aplicando o modelo como mapeamento de x para y

plt.figure(1)
plt.subplot(1,1,1)
plt.scatter(x, t, color='b', facecolors='none', marker = 'o')
plt.plot(x,y, color='r')
plt.ylim(-2,2)
plt.xlim(0,1)  
