# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from matplotlib import colors

cmap = colors.LinearSegmentedColormap('red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

iris = ds.load_iris()
X = iris.data  # features
t = iris.target
t_names = iris.target_names

X1 = np.asarray(X[t == 0,:])
X2 = np.asarray(X[t == 1,:])
X3 = np.asarray(X[t == 2,:])

X_train = np.concatenate([X1[:25,:],X2[:25,:]])
# t_train = np.concatenate([np.asarray(t[:25]), np.asarray(t[50:75]), np.asarray(t[125:])])
t_train = np.concatenate([np.zeros(25, dtype=int), np.ones(25, dtype=int)])

X_test = np.concatenate([X1[25:,:],X2[25:,:]])

#m1 = np.asarray([np.mean(X_train[t_train == 0,i]) for i in range(0,4)])
m1 = np.mean(X_train[t_train == 0,:], axis=0)

#m2 = np.asarray([np.mean(X_train[:,i]) for i in range(0,4)])
m2 = np.mean(X_train[t_train == 1,:], axis=0)

S1 = np.zeros((4,4)) # Matriz de dispersão para classe 1
for xi in X_train[t_train == 0]: S1 += (xi.reshape(4,1)-m1).dot((xi.reshape(4,1)-m1).T)
# S1 = (X_train[t == 0,:].T @ X_train[t == 0,:]) - (m1.T @ m1) 

S2 = np.zeros((4,4)) # Matriz de dispersão para classe 1
for xi in X_train[t_train == 1]: S2 += (xi.reshape(4,1)-m2).dot((xi.reshape(4,1)-m2).T)
# S1 = (X_train[t == 1,:].T @ X_train[t == 1,:]) - (m2.T @ m2)   

SW = S1 + S2

w = np.linalg.inv(SW) @ (m2-m1)

print(w)

y_train = X_train @ w
y_test = X_test @ w

y_test_pred = np.where(y_test >= 0, 1, 0)

#limiar = ((m1 * m2) / 2) @ w
#y_test_cl = (y_test > limiar).astype(np.int)

plt.figure(figsize=(10,7))
for i, name in zip([0, 1], [t_names[0],t_names[1]]):
    plt.scatter(X_test[y_test_pred == i,0], X_test[y_test_pred == i,2], alpha=0.7, label=name)

# Fronteira de decisão e bgcolor
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
P = np.c_[xx.ravel(), xx.ravel(), yy.ravel(), yy.ravel()]
Z = P @ w
Z = np.where(Z >= 0, 1, 0)
Z = np.c_[Z,np.where(Z >= 0, 0, 1)]
Z = Z[:,1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes', norm=colors.Normalize(0., 1.), zorder=0)
plt.contour(xx, yy, Z, [0], linewidths=2., colors='white')

plt.legend(loc='best', scatterpoints=1)
#plt.savefig('figures/Fig4.3.1_.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
# plt.plot(y_test1, y_test1, c='k')



### Opção 2
#Sa2 = np.cov(Xa_train, rowvar=False) # rowvar=False assume que os exemplos estao nas linhas
#Sb2 = np.cov(Xb_train, rowvar=False)
#S2 = Sa2 + Sb2 # cov_inclass
#w2 = np.linalg.solve(S2, m_a - m_b) # ~ np.linalg.inv(S2) @ (m_a - m_b)
#w2 /= np.linalg.norm(w2).clip(min=1e-10)
#y_train2 = X_train @ w2
#y_test2 = X_test @ w2
#plt.figure(figsize=(10,7))
#plt.plot(y_test2)


### Opção 3 - Sklearn
#lda = LinearDiscriminantAnalysis()
#lda.fit(X_train, t_train)
#lda_scores = lda.transform(X_test)
#lda_pred = lda.predict(X_test)
#plt.figure()
#plt.plot(lda_scores, np.zeros(50), c='r') #lda_scores[t_train == i, 0])
#for i, name in zip([0, 1], [t_names[1],t_names[2]]):
#    # plt.scatter(lda_scores[t_train == i, 0], lda_scores[t_train == i, 0], alpha=.8, label=name)
#    plt.scatter(X_test[t_train == i,3], lda_scores[t_train == i, 0], alpha=.7)
#    #plt.plot(lda_scores[t_train == i, 0], np.zeros(25)) #lda_scores[t_train == i, 0])