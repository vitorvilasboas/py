# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#gerar curva contnua através dos pontos de cordenada (x,y)
plt.figure(figsize=(8,5))
x = np.linspace(0,1,100)
y = np.sin(2*np.pi*x) # f(x)
plt.plot(x,y, 'g')

x1 = np.linspace(0,1,10)
ruido = np.random.randn(len(x1))
t = np.sin(2*np.pi*x1) + 0.1*ruido
#t = np.sin(2*np.pi*x1) + choice(norm(x,0.3).args)
plt.scatter(x1,t, color='b')

A = np.asarray([x1**i for i in range(9+1)]).T
w = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T,t)) # calculo direto
y = np.dot(A,w)
#plt.plot(x1,y)

x2 = np.linspace(0,1,10000)
A1 = np.asarray([x2**i for i in range(9+1)]).T
y2 = np.dot(A1,w)
plt.plot(x2,y2, 'r')



## testando...
x_train = np.linspace(0, 1, 10)
noise = np.random.normal(scale=0.3, size=x_train.shape)
y_train = np.sin(2 * np.pi * x_train) + noise #scale=std

x_test = np.linspace(0, 1, 100)
y_test = np.sin(2 * np.pi * x_test)

plt.figure()
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.legend()
