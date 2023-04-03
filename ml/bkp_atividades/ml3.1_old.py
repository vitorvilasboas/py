# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881])
t = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839])
M = 3

s = 2 # tem interpretação de variancia (pode ser calculada a partir dos blocos de x que calculam cada mu)

#mu = np.random.randint(len(x), size=M)
pos = np.random.randint(len(x)-1, size=M)
mu = np.sort([x[i] for i in pos], ) # 1) gerar números aleatórios dentro da faixa de valores de x
#mu = np.array([np.mean(x[0:3]), np.mean(x[3:6]), np.mean(x[6:])]) # 2) separar x em 3 e obter a média de x para cada região..
# outra opção é adotar a qtd de mu igual à N... assim haverá uma gaussiana vcentrada em cada elemento de x
# para determinar os centros pode-se utilizar o algoritmo o k-means

A = np.zeros([len(x),M+1])
A[:,0] = np.ones([len(x)])

for i in range(M): A[:,i+1] = np.asarray([(np.exp((-1)*(((xi - mu[i])**2)/2*(s**2)))).T for xi in x]).T

w = inv(A.T.dot(A)).dot(A.T.dot(t))
y = A.dot(w) # aplicando o modelo como mapeamento de x para y


plt.figure()
plt.scatter(x, t, facecolor="none", edgecolor="b", s=50, label="target train")
plt.scatter(x, y, facecolor="none", edgecolor="r", s=50, label="predicted train")
plt.plot(x, y, c="r", label="predicted test $y(x,w)$")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()


plt.figure(figsize=(10,6))
plt.scatter(x, t, facecolor="none", edgecolor="b", s=50, label="target train")
plt.plot(x_test, y_test, c="r", label="$y(x,w)$")
plt.plot(x_correct, t_correct, c="g", alpha=0.6, label="$\sin(2\pi x)$")
plt.legend()
plt.ylim(-6,6)
