# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class Regression(object):
    def __init__(self, lambd:float=1., beta:float=1.):
        self.lambd = lambd
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def fit(self, X:np.ndarray, t:np.ndarray):
        self.w_precision = self.lambd * np.eye(np.size(X, 1)) + self.beta * X.T @ X
        self.w_mean = np.linalg.solve(self.w_precision, self.lambd * np.eye(np.size(X, 1)) @ np.zeros(np.size(X, 1)) + self.beta * X.T @ t )
        self.w_cov = np.linalg.inv(self.w_precision)

    def predict(self, X:np.ndarray, return_std:bool=False, sample_size:int=None):
        y = X @ self.w_mean
        return y


x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

#x_train = np.linspace(0.01,1,25, endpoint=False)
#t_train = np.sin(2*np.pi*x_train) + np.random.normal(scale=0.1, size=x_train.shape) #scale=desvio padrão # 0.1 * np.random.randn(len(x_train))

x_test = np.linspace(0.01,0.99,100, endpoint=False)
t_test = np.sin(2 * np.pi * x_test) + np.random.normal(scale=0.1, size=x_test.shape) #scale=desvio padrão # 0.1 * np.random.randn(len(x_test))

M = 24

mu = np.linspace(x_train[0], x_train[-1], M) 
mu = mu[:, None] 
var = np.var(x_train) 

# np.exp(-18) == 1e-8       # 1e-3 == 0.001 ~ np.exp(-6.9)# 
ln_lambd = [2.6, -0.31, -2.4] 

beta = 1.

for l in ln_lambd:
    y_list = []
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    for i in range(100):
        
        x_train = np.linspace(0.01,1,25, endpoint=False)
        #np.random.shuffle(x_train)
        t_train = np.sin(2 * np.pi * x_train) + np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão # 0.1 * np.random.randn(len(x_train))
        
        x_train1 = x_train[:, None]
        base = [np.ones(len(x_train1))]
        for m in mu: base.append(np.exp(-0.5 * np.sum(np.square(x_train1 - m), axis=-1) / var))
        A_train = np.asarray(base).T
        
        x_test1 = x_test[:, None]     
        base = [np.ones(len(x_test1))]
        for m in mu: base.append(np.exp(-0.5 * np.sum(np.square(x_test1 - m), axis=-1) / var))
        A_test = np.asarray(base).T
           
#        model = Regression(alpha=np.exp(l), beta=beta)
#        model.fit(A_train, t_train)
#        y_test = model.predict(A_test)
        
        w_precision = np.exp(l) * np.eye(M+1) + beta * A_train.T @ A_train
        w_mean = np.linalg.solve( w_precision, np.exp(l) * np.eye(M+1) @ np.zeros(M+1) + beta * A_train.T @ t_train ) # w ~ N(w|0, lambd^(-1)I)     t ~ N(t|X @ w, beta^(-1))
        y_test = A_test @ w_mean # - mean of the predictive distribution

#        w_cov = np.linalg.inv(w_precision)
#        w_sample = np.random.multivariate_normal(w_mean, w_cov, size=20) # size = number of samples to draw from the predictive distribution
#        y_sample = A_test @ w_sample.T # samples from the predictive distribution, 
#        y_var = 1 / beta + np.sum(A_test @ w_cov * A_test, axis=1)
#        y_std = np.sqrt(y_var) # standard deviation of the predictive distribution
        
        y_list.append(y_test)
        
        if i < 20: plt.plot(x_test, y_test, c="r") 
    
    plt.annotate(('ln $\lambda$ = {}'.format(l)), xy=(0.8, 1))
    plt.ylabel('$t$', fontsize=13, rotation = 360)
    plt.xlabel('$\mathit{x}$', fontsize=13,labelpad=-15)
    plt.xticks([0,1])
    plt.yticks([-1,0,1])
    plt.ylim(-1.5,1.5)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_correct, t_correct, label="$\sin(2\pi x)$")
    plt.plot(x_test, np.asarray(y_list).mean(axis=0), c='r', label="prediction")
    plt.ylabel('$t$', fontsize=13, rotation = 360)
    plt.xlabel('$\mathit{x}$', fontsize=13,labelpad=-15)
    plt.xticks([0,1])
    plt.yticks([-1,0,1])
    plt.ylim(-1.5, 1.5)
    plt.legend()
    
    plt.savefig(('figures/Fig3.5_{}.png'.format(l)), format='png', dpi=300, transparent=True, bbox_inches='tight')


