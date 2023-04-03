# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds

iris = ds.load_iris()
X = iris.data  # features
t = iris.target
t_names = iris.target_names

X1 = np.asarray(X[t == 0,:])
X2 = np.asarray(X[t == 1,:])
X3 = np.asarray(X[t == 2,:])

X_train = np.concatenate([X1[:25,:],X3[:25,:]])
# t_train = np.concatenate([np.asarray(t[:25]), np.asarray(t[50:75]), np.asarray(t[125:])])
t_train = np.concatenate([np.zeros(25, dtype=int), np.ones(25, dtype=int)])

Xa_train = X_train[t_train == 0]
Xb_train = X_train[t_train == 1]

Xa_test = X1[25:,:]
Xb_test = X3[25:,:]
X_test = np.concatenate([Xa_test,Xb_test])

#m_a = np.asarray([np.mean(Xa_train[:,i]) for i in range(0,4)])
m_a = np.mean(Xa_train, axis=0)

#m_b = np.asarray([np.mean(Xb_train[:,i]) for i in range(0,4)])
m_b = np.mean(Xb_train, axis=0)

### Opção 1
Sa1 = (Xa_train.T @ Xa_train) - (m_a.T @ m_a) # (c1.T).dot(c1) - (m1.T).dot(m1)

Sb1 = (Xb_train.T @ Xb_train) - (m_b.T @ m_b) # (c2.T).dot(c2) - (m2.T).dot(m2)

S1 = Sa1 + Sb1 # cov_inclass

w1 = np.linalg.inv(S1) @ (m_a - m_b)

y_train1 = X_train @ w1
y_test1 = X_test @ w1

limiar = 0
y_test_cl = (y_test1 > limiar).astype(np.int)

x_plot = 3
plt.figure(figsize=(10,7))
for i, name in zip([0, 1], [t_names[0],t_names[2]]):
    plt.scatter(X_test[t_train == i,x_plot], y_test1[t_train == i], alpha=0.7, label=name)
x_ = np.linspace(min(X_test[:,x_plot]), max(X_test[:,x_plot]), 100) # np.linspace(0.8,2.6,1000)
y_ = np.linspace(min(y_test1), max(y_test1), 100) # np.linspace(-0.05,0.08,1000)
plt.plot(x_,y_, c='r')
plt.legend(loc='best', scatterpoints=1)
plt.savefig('figures/Fig4.3.1.2.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

# plt.plot(y_test1, y_test1, c='k')