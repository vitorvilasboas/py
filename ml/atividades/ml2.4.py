# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

x_test = np.linspace(0.1,0.9,1000, endpoint=False)
t_test = np.sin(2*np.pi*x_test) + 0.1 * np.random.randn(len(x_test))
# t_test = np.sin(2*np.pi*x_test) + np.random.normal(scale=0.3, size=x_train.shape)

M = 3
l = [-18]
A_train = np.asarray([x_train**i for i in range(M+1)]).T
A_test = np.asarray([x_test**i for i in range(M+1)]).T

# Mínimos quadrados
w = np.linalg.solve(np.exp(l) * np.identity(M+1) + A_train.T @ A_train, A_train.T @ t_train)  
y_train = A_train @ w 
y_test = A_test @ w  
Erms = np.sqrt(np.mean(np.square(y_train-t_train))) #np.sqrt((y_train-t_train).T.dot(y_train-t_train)/len(x_train))
ErmsV = np.sqrt(np.mean(np.square(y_test-t_test))) #np.sqrt((y_test-t_test).T.dot(y_test-t_test)/len(x_test))

# Máxima Verossimilhança
alpha = np.exp(l)
beta = 1  
beta_ml = alpha * np.identity(M+1) + beta * A_train.T @ A_train # np.eye(np.size(A_train, 1))
w_ml = np.linalg.solve(beta_ml, ( alpha * np.identity(M+1) ) @ np.zeros(M+1) + beta * A_train.T @ t_train) # np.zeros(np.size(A_train, 1))
y_train = A_train @ w_ml
y_test = A_test @ w_ml
Err_train = np.sqrt(1 / beta + np.sum(A_train @ np.linalg.inv(beta_ml) * A_train, axis=1)) # y_std = np.sqrt(y_var)
Err_test = np.sqrt(1 / beta + np.sum(A_test @ np.linalg.inv(beta_ml) * A_test, axis=1)) # inversa de beta, inverso da variancia(covariancia)

plt.subplots(1, figsize=(10,7)) 

sup_y = np.linspace(-1.5,1.5,100)
sup_x = np.ones(len(sup_y)) * x_test[400]
plt.plot(sup_x,sup_y, c="k")

pont_x = np.linspace(-0.5,x_test[400],100)
pont_y = np.ones(len(pont_x)) * y_test[400]
plt.plot(pont_x,pont_y, c="lime", ls='-.')

plt.plot(x_test, y_test, c="r", label="$y(x,w)$")
plt.scatter(x_test[400],y_test[400], c = "lime")


N = norm.pdf(np.linspace(-Err_test[400]-25,Err_test[400]+14,1000), y_test[400], Err_test[400])
#N = (1/np.sqrt(2*np.pi*np.var(Err_test))) * np.exp((-1/(2*np.var(Err_test)))*((Err_test-Err_test[400])**2)  )
plt.plot(N*5**(-1)+x_test[400], np.linspace(-1.5,1.5,len(y_test)), c="b", label="$p(t|x_{400},\mathbf{w},\beta)$")


plt.ylabel('$\mathit{t}$', fontsize=13, rotation = 360)
plt.xlabel('$\mathit{x}$', fontsize=13)
plt.xticks([x_test[400]])
plt.yticks([y_test[400]])
plt.ylim(-1.2,1.2)
plt.xlim(0,1)
    
plt.legend(loc='lower left')
# plt.savefig('figures/Fig2.4.1_.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
