# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Modelo de Regressão Bayesiano
# w ~ N(w|0, lambd^(-1) * I)
# y = A @ w
# t ~ N(t|A @ w, beta^(-1))

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ]) # variável dependente treinamento

x_test = np.linspace(0,1,100, endpoint=False) #0.1-0.8
t_test = np.sin(2*np.pi*x_test) + 0.1 * np.random.randn(len(x_test)) 

M = 9
A_train = np.asarray([x_train**i for i in range(M+1)]).T # variável independente treinamento
A_test = np.asarray([x_test**i for i in range(M+1)]).T # variável independente teste

lambd = 2e-3
beta = 2

ndim = np.size(A_train, 1)
mean_prev, precision_prev = np.zeros(ndim), lambd * np.eye(ndim)
w_precision = precision_prev + beta * A_train.T @ A_train
w_mean = np.linalg.solve( w_precision, precision_prev @ mean_prev + beta * A_train.T @ t_train )
w_cov = np.linalg.inv(w_precision)

# Calculando a média e o desvio padrão da distribuição preditiva (dp)
# w_sample = np.random.multivariate_normal(w_mean, w_cov, size=sample_size ) # if sample_size is not None: - número de amostras a serem retiradas da distribuição preditiva
# y_sample = A_test @ w_sample.T   # amostras da dp
y = A_test @ w_mean # média da dp
y_var = 1 / beta + np.sum(A_test @ w_cov * A_test, axis=1) 
y_err = np.sqrt(y_var) # y_std = y_err  # desvio padrao da dp

plt.figure(figsize=(8,5))
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, t_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y, c="r", label="mean")
plt.fill_between(x_test, y - y_err, y + y_err, color="pink", label="std.", alpha=0.5)
plt.xlim(-0.1, 1.1)
plt.ylim(-1.5, 1.5)
plt.annotate("M=9", xy=(0.8, 1))
plt.legend(bbox_to_anchor=(1.05, 1.), loc='best', borderaxespad=0.)


#x_test = np.arange(-10, 10, 0.001)
y2 = norm.pdf(x_test,y,y_err)
plt.plot(y2, x_test)
#plt.fill_between(y2,x_test,0, alpha=0.3, color='b')