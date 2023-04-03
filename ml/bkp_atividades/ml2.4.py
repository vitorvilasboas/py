#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
import scipy.stats
import math

class PolynomialFeature(object):
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree # degree : int - degree of polynomial

    def transform(self, x):
        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()
    
class Regression(object):
    pass

class BayesianRegression(Regression):
    def __init__(self, alpha:float=1., beta:float=1.):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _is_prior_defined(self) -> bool:
        return self.w_mean is not None and self.w_precision is not None

    def _get_prior(self, ndim:int) -> tuple:
        if self._is_prior_defined():
            return self.w_mean, self.w_precision
        else:
            return np.zeros(ndim), self.alpha * np.eye(ndim)
    
    def fit(self, X:np.ndarray, t:np.ndarray):
        mean_prev, precision_prev = self._get_prior(np.size(X, 1))

        w_precision = precision_prev + self.beta * X.T @ X
        w_mean = np.linalg.solve(
            w_precision,
            precision_prev @ mean_prev + self.beta * X.T @ t
        )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)
    
    def predict(self, X:np.ndarray, return_std:bool=False, sample_size:int=None):
        if sample_size is not None:
            w_sample = np.random.multivariate_normal(
                self.w_mean, self.w_cov, size=sample_size
            )
            y_sample = X @ w_sample.T
            return y_sample
        y = X @ self.w_mean
        if return_std:
            y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y



if __name__ == "__init__":
    x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
    t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])
    
    x_correct = np.linspace(0, 1, 100)
    t_correct = np.sin(2 * np.pi * x_correct)
    
    x_test = np.linspace(0.1,0.9,1000, endpoint=False)
    noise = np.random.randn(len(x_test))
    # noise = np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
    t_test = np.sin(2*np.pi*x_test) + 0.1 * noise 
    
    feature = PolynomialFeature(9)
    A_train = feature.transform(x_train)
    A_test = feature.transform(x_test)
    
    
    
    model = BayesianRegression(alpha=np.exp(-18), beta=2)
    model.fit(A_train, t_train)
    y_train, y_train_err = model.predict(A_train, return_std=True)
    y_test, y_test_err = model.predict(A_test, return_std=True)
    
    plt.plot(x_test[50]*np.ones(len(x_test)),np.linspace(-2,2,len(x_test)), c="k")
    
    plt.plot(np.linspace(-1,x_test[50],len(x_test)),y_test[50]*np.ones(len(x_test)), c="y", linestyle='--')
    
    plt.scatter(x_test[50], y_test[50], c='r')
    
    teste1 = np.random.normal(scale=0.3, size=len(x_test)) + x_test[50]
    teste = np.linspace(-y_test_err[50]-x_test[50],y_test_err[50]+x_test[50],100)
    
    # N = np.random.normal(y_train[5],y_train_err[5],len(x_train))
    
    N1 = scipy.stats.norm.pdf(teste,y_test[50],y_test_err[50])
    
    plt.plot(N1,np.linspace(-y_test_err[50],y_test_err[50],len(x_test)))
    
    
    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y_test, c="r", label="mean")
    #plt.fill_between(x_test, y_test - y_test_err, y_test + y_test_err, color="pink", label="std.", alpha=0.5)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1.5, 1.5)
    plt.annotate("M=9", xy=(0.8, 1))
    plt.legend(bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0.)
    plt.show()