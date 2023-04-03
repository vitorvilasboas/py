import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

base = pd.read_csv('iris.csv')
features = base.iloc[:, 0:4].values
saidas = base.iloc[:, 4].values

c1 = np.asarray(features[0:40,:])

c2 = np.asarray(features[60:100,:])

m1 = np.asarray([np.mean(c1[:,i]) for i in range(0,4)])

m2 = np.asarray([np.mean(c2[:,i]) for i in range(0,4)])

s1 = (c1.T).dot(c1) - (m1.T).dot(m1)

s2 = (c2.T).dot(c2) - (m2.T).dot(m2)

S = s1 + s2

w = inv(S).dot(m2-m1)

y = (features[40:60,:]).dot(w.T)

plt.plot(y)

