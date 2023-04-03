# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from prml.rv.gaussian import Gaussian

class FishersLinearDiscriminant(object):
    def __init__(self, w:np.ndarray=None, threshold:float=None):
        self.w = w
        self.threshold = threshold

    def fit(self, X:np.ndarray, t:np.ndarray):
        """ estimate parameter given training dataset
            X : (N, D) np.ndarray, training dataset independent variable
            t : (N,) np.ndarray, training dataset dependent variable binary 0 or 1 """
        X0 = X[t == 0]
        X1 = X[t == 1]
        m0 = np.mean(X0, axis=0)
        m1 = np.mean(X1, axis=0)
        cov_inclass = np.cov(X0, rowvar=False) + np.cov(X1, rowvar=False)
        self.w = np.linalg.solve(cov_inclass, m1 - m0)
        self.w /= np.linalg.norm(self.w).clip(min=1e-10)

        g0 = Gaussian()
        g0.fit((X0 @ self.w))
        g1 = Gaussian()
        g1.fit((X1 @ self.w))
        root = np.roots([ g1.var - g0.var, 2 * (g0.var * g1.mu - g1.var * g0.mu),
            g1.var * g0.mu ** 2 - g0.var * g1.mu ** 2 - g1.var * g0.var * np.log(g1.var / g0.var) ])
        if g0.mu < root[0] < g1.mu or g1.mu < root[0] < g0.mu: self.threshold = root[0]
        else: self.threshold = root[1]

    def transform(self, X:np.ndarray):
        return X @ self.w

    def classify(self, X:np.ndarray):
        return (X @ self.w > self.threshold).astype(np.int)


iris = ds.load_iris()
X = iris.data  # features
y = iris.target
y_names = iris.target_names

pos = np.nonzero(y)
c1 = np.asarray(X[0:25,:])
c2 = np.asarray(X[75:100,:])
c3 = np.asarray(X[125:,:])

#c = c1.T.dot(c1) # soma dos quadrados
#ca = c1.dot(c1.T)
#z = np.cov(c1.T)
#teste = sum(c1**2)
#variance = np.var(c1[:,0]) # ~ sum([(c1[i,0] - np.mean(c1[:,0]))**2 for i in range(len(c1[:,0]))])/len(c1[:,0])
#std = np.std(c1[:,0]) # np.sqrt(variance)
#var_all_feat = [np.var(c1[:,i]) for i in range(len(c1.T))]

x_train = np.concatenate([c2,c3])
y_train = np.concatenate([np.asarray(y[50:75]), np.asarray(y[125:])])
m2 = np.asarray([np.mean(c2[:,i]) for i in range(0,4)])
m3 = np.asarray([np.mean(c3[:,i]) for i in range(0,4)])
s2 = c2.T @ c2 - m2.T @ m2 # (c1.T).dot(c1) - (m1.T).dot(m1)
s3 = c3.T @ c3 - m3.T @ m3 # (c2.T).dot(c2) - (m2.T).dot(m2)
S = s2 + s3
w = np.linalg.inv(S) @ (m3-m2)
yt = x_train @ w.T
yv = X[75:125,:] @ w.T
#plt.plot(yv)
