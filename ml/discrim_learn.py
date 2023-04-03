# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds

iris = ds.load_iris()
X = iris.data  # features
y = iris.target
y_names = iris.target_names


c1 = np.asarray(X[y == 0,:])
c2 = np.asarray(X[y == 1,:])
c3 = np.asarray(X[y == 2,:])

x_train = np.concatenate([c2[:25,:],c3[:25,:]])
t_train = np.concatenate([np.asarray(y[:25]), np.asarray(y[50:75])])

x1_test = c2[25:,:]
x2_test = c3[25:,:]
x_test = np.concatenate([x1_test,x2_test])


X0 = x_train[t_train == 0]
X1 = x_train[t_train == 1]

m0 = np.mean(X0, axis=0)
m1 = np.mean(X1, axis=0)

cov_inclass = np.cov(X0, rowvar=False) + np.cov(X1, rowvar=False)

wa = np.linalg.solve(cov_inclass, m1 - m0)

wa /= np.linalg.norm(wa).clip(min=1e-10)

y_test = x_test @ wa


limiar = 1
y_test_cl = (y_test > limiar).astype(np.int)

plt.figure(figsize=(10,7))
plt.plot(y_test)


## Least squares
#t_train_2 = np.eye(np.max(t_train) + 1)[t_train]
#W = np.linalg.pinv(x_train) @ t_train_2
#y_test = np.argmax(x_test @ W, axis=-1)

## Logistic Regression
#w = np.zeros(np.size(x_train, 1))
#for _ in range(100):
#    w_prev = np.copy(w)
#    y_train = np.tanh(x_train @ w * 0.5) * 0.5 + 0.5 
#    grad = x_train.T @ (y_train - t_train)
#    hessian = (x_train.T * y_train * (1 - y_train)) @ x_train
#    try:
#        w -= np.linalg.solve(hessian, grad)
#    except np.linalg.LinAlgError:
#        break
#    if np.allclose(w, w_prev):
#        break
#threshold = 0.5  
#y_test2 = (np.tanh(x_test @ w * 0.5) * 0.5 + 0.5 > threshold).astype(np.int)

#plt.subplot(1, 2, 1)
#plt.scatter(x_train, x_train, c=y_train)
#plt.contourf(x1_test, x2_test, y_test.reshape(25, 2), alpha=0.2, levels=np.linspace(0, 1, 3))
#plt.xlim(-5, 15)
#plt.ylim(-5, 15)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.title("Least Squares")

#plt.subplot(1, 2, 2)
#plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
#plt.contourf(x1_test, x2_test, y_test2.reshape(25, 2), alpha=0.2, levels=np.linspace(0, 1, 3))
#plt.xlim(-5, 15)
#plt.ylim(-5, 15)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.title("Logistic Regression")