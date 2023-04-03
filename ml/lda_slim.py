import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
X = ds.load_iris().data
t = ds.load_iris().target
ma = np.mean(X[t == 0,:], axis=0)
mb = np.mean(X[t == 1,:], axis=0)
Sa = np.zeros((4,4))
Sb = np.zeros((4,4))
for xj in X[t == 0]: Sa += (xj.reshape(4,1)-ma).dot((xj.reshape(4,1)-ma).T)
for xj in X[t == 1]: Sb += (xj.reshape(4,1)-mb).dot((xj.reshape(4,1)-mb).T)  
Sw = Sa + Sb
w1 = np.linalg.inv(Sw) @ (mb-ma)
y1 = X @ w1
#plt.figure(figsize=(10,7))
#plt.scatter(X[t == 0,1], y[t == 0], alpha=0.7)
#plt.legend(loc='best', scatterpoints=1)