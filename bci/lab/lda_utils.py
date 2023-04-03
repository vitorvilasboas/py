# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = ds.load_iris()
X = iris.data  # features
y = iris.target

pos = np.nonzero(y)


c1 = np.asarray(X[0:25,:])

c2 = np.asarray(X[75:100,:])

c3 = np.asarray(X[125:,:])

c = c1.T @ c1

teste = sum(c1**2)

variance = np.var(c1[:,0]) # ~ sum([(c1[i,0] - np.mean(c1[:,0]))**2 for i in range(len(c1[:,0]))])/len(c1[:,0])

x_train = np.concatenate([c1,c2])

y_train = np.concatenate([np.asarray(y[0:25]), np.asarray(y[75:100])])

m1 = np.asarray([np.mean(c1[:,i]) for i in range(0,4)])

m2 = np.asarray([np.mean(c2[:,i]) for i in range(0,4)])

s1 = c1.T @ c1 - m1.T @ m1 # (c1.T).dot(c1) - (m1.T).dot(m1)

s2 = c2.T @ c2 - m2.T @ m2 # (c2.T).dot(c2) - (m2.T).dot(m2)

S = s1 + s2

w = np.linalg.inv(S) @ (m2-m1)

yt = X[0:100,:] @ w.T

yv = X[25:75,:] @ w.T

plt.plot(yv)



lda = LinearDiscriminantAnalysis()

lda.fit(x_train, y_train)

scores = lda.predict(np.asarray(X[25:75,:])) #.transform(X)

plt.figure(figsize=(10,7))
plt.plot(scores)


