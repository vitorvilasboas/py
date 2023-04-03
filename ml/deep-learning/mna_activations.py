# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:05:03 2019

@author: Rafael Rocha
"""

#%% Pacotes utilizados
import numpy as np
import matplotlib.pyplot as plt

#%% Funções
def sigmoid_func(x, a=1):
    y = 1/(1+np.exp(-x*a))
    dy = y*(1-y)
    return y, dy

def tanh_func(x):    
    y = np.sinh(x)/np.cosh(x)
    dy = 1 - np.power(y,2)
    return y, dy

def relu_func(x):
    y = np.maximum(x,0)
    dy = np.heaviside(x,x)
    return y, dy

#%% Variáveis
x = np.linspace(-10,10)

#%% Função Sigmoid
"""
a = 1, a = 2 e a = 3
"""
y, dy = sigmoid_func(x, a=32)

plt.figure()
plt.plot(x, y)
plt.plot(x, dy)
plt.ylabel(r'$\varphi$(v)')
plt.xlabel('v')
plt.title('Sigmoide')
plt.axvline(0, color='black')
plt.grid()

#%% Função arco tangente
y, dy = tanh_func(x)

plt.figure()
plt.plot(x, y)
plt.plot(x, dy)
plt.xlabel('v')
plt.ylabel(r'$\varphi$(v)')
plt.title('Tanh')
plt.grid()

#%% Função ReLU
y, dy = relu_func(x)

plt.figure()
plt.plot(x, y)
plt.plot(x, dy)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('ReLU')
plt.grid()