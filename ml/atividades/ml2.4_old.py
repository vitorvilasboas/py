#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

class BayesianRegression(object):
    def __init__(self, alpha:float=1., beta:float=1.):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _is_prior_defined(self) -> bool:
        return self.w_mean is not None and self.w_precision is not None

    def _get_prior(self, ndim:int) -> tuple:
        if self.w_mean is not None and self.w_precision is not None: # if self._is_prior_defined():
            print('Sim')
            return self.w_mean, self.w_precision
        else: 
            print('Não')
            return np.zeros(ndim), self.alpha * np.eye(ndim)
    
    def fit(self, X:np.ndarray, t:np.ndarray):
        ndim = np.size(X, 1)
        mean_prev, precision_prev = np.zeros(ndim), self.alpha * np.eye(ndim) # self._get_prior(np.size(X, 1))
        w_precision = precision_prev + self.beta * X.T @ X
        w_mean = np.linalg.solve( w_precision, precision_prev @ mean_prev + self.beta * X.T @ t )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)
    
    def predict(self, X:np.ndarray, return_std:bool=False, sample_size:int=None):
        if sample_size is not None:
            print('Sim')
            w_sample = np.random.multivariate_normal( self.w_mean, self.w_cov, size=sample_size )
            y_sample = X @ w_sample.T
            return y_sample
        y = X @ self.w_mean
        if return_std:
            print('Não')
            y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y

if __name__ == "__init__":
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
    w = np.linalg.solve(np.exp(l) * np.identity(M+1) + A_train.T @ A_train, A_train.T @ t_train)
    y_train = A_train @ w 
    y_test = A_test @ w
    Erms = (np.sqrt(np.mean(np.square(y_train - t_train)))) #~ np.sqrt((y_train-t_train).T.dot(y_train-t_train)/len(x_train))
    ErmsV = (np.sqrt(np.mean(np.square(y_test - t_test)))) #~ np.sqrt((y_test-t_test).T.dot(y_test-t_test)/len(x_test))
   
    
    alpha = np.exp(-18)
    beta = 1
    ndim = np.size(A_train, 1)
    mean_prev, precision_prev = np.zeros(ndim), alpha * np.eye(ndim) # self._get_prior(np.size(X, 1))
    w_precision = precision_prev + beta * A_train.T @ A_train
    w_mean = np.linalg.solve( w_precision, precision_prev @ mean_prev + beta * A_train.T @ t_train )
    w_cov = np.linalg.inv(w_precision) # inversa de beta
    
    y_train = A_train @ w_mean
    y_test = A_test @ w_mean
    
    y_train_err = np.sqrt(1 / beta + np.sum(A_train @ w_cov * A_train, axis=1)) # y_std = np.sqrt(y_var)
    
    y_test_err = np.sqrt(1 / beta + np.sum(A_test @ w_cov * A_test, axis=1))
    
    
    
#    model = BayesianRegression(alpha=np.exp(-18), beta=1)
#    model.fit(A_train, t_train)
#    y_train, y_train_err = model.predict(A_train, return_std=True)
#    y_test, y_test_err = model.predict(A_test, return_std=True)
    
    
    
#    
#    
#    
#    plt.plot(x_test[50]*np.ones(len(x_test)),np.linspace(-2,2,len(x_test)), c="k")
#    
#    plt.plot(np.linspace(-1,x_test[50],len(x_test)),y_test[50]*np.ones(len(x_test)), c="y", linestyle='--')
#    
#    plt.scatter(x_test[50], y_test[50], c='r')
#    
#    teste1 = np.random.normal(scale=0.3, size=len(x_test)) + x_test[50]
#    teste = np.linspace(-y_test_err[50]-x_test[50],y_test_err[50]+x_test[50],100)
#    
#    # N = np.random.normal(y_train[5],y_train_err[5],len(x_train))
#    
#    N1 = scipy.stats.norm.pdf(teste,y_test[50],y_test_err[50])
#    
#    plt.plot(N1,np.linspace(-y_test_err[50],y_test_err[50],len(x_test)))
#    
#    
#    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
#    plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
#    plt.plot(x_test, y_test, c="r", label="mean")
#    #plt.fill_between(x_test, y_test - y_test_err, y_test + y_test_err, color="pink", label="std.", alpha=0.5)
#    plt.xlim(-0.1, 1.1)
#    plt.ylim(-1.5, 1.5)
#    plt.annotate("M=9", xy=(0.8, 1))
#    plt.legend(bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0.)