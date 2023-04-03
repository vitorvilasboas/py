# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np
import matplotlib.pyplot as plt
from prml.linear import FishersLinearDiscriminant
import sklearn.datasets as ds
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from matplotlib import colors

cmap = colors.LinearSegmentedColormap('red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

#from scipy.linalg import eigh

iris = ds.load_iris()
X = iris.data  # features
t = iris.target
t_names = iris.target_names

n_cls = 3

Xa = np.asarray(X[t == 0,:])
Xb = np.asarray(X[t == 1,:])
Xc = np.asarray(X[t == 2,:])

Xa_train = Xa[:25,:]
Xb_train = Xb[:25,:]
Xc_train = Xc[:25,:]

Xa_test = Xa[25:,:]
Xb_test = Xb[25:,:]
Xc_test = Xc[25:,:]

X_train = np.concatenate([Xa_train,Xb_train,Xc_train])
t_train = np.concatenate([t[:25], t[50:75], t[100:125]])

X_test = np.concatenate([Xa_test,Xb_test,Xc_test])

# A padronização de um conjunto de dados é um requisito comum para muitos estimadores 
# de aprendizado de máquina: eles podem se comportar mal se os recursos individuais não 
# se parecerem mais ou menos com dados padrão normalmente distribuídos
# Muitos elementos usados ​​na função objetivo de um algoritmo de aprendizado (como o kernel RBF 
# do Support Vector Machines ou os regularizadores L1 e L2 dos modelos lineares) assumem que 
# todos os recursos estão centralizados em torno de 0 e têm variação na mesma ordem. 
# Se um recurso tem uma variação que é ordens de magnitude maior que os outros, ele pode 
# dominar a função objetivo e tornar o estimador incapaz de aprender com outros recursos 
# corretamente, conforme o esperado.
# X_train_norm ~ score z = (x-u)/s normaliza/transforma os dados a partir da média (u) e o desvio_padrao (s)
X_train_norm = StandardScaler().fit_transform(X_train)
X_test_norm = StandardScaler().fit_transform(X_train)

SW = np.zeros((len(X[0]),len(X[0]))) # Variancia intra-classes
for i in range(n_cls): SW += np.cov(X_train[t_train == i].T) # to 3 classes

N = np.bincount(t_train) # vetor com n_amostras de cada classe no conjuntos de treinamento

m_k = [ np.mean(X_train_norm[t_train == i], axis=0) for i in range(n_cls) ] # média em cada k classes

m_full = np.mean(X_train_norm, axis=0) # média geral

SB = np.zeros((len(X[0]),len(X[0]))) # Variancia entre-classe
for i in range(n_cls): SB += N[i] * (((m_k[i]-m_full).reshape(4,1)).dot((m_k[i]-m_full).reshape(1,4)))

[D, W] = np.linalg.eig(np.linalg.inv(SW).dot(SB))

Y_train = X_train_norm.dot(W[:,0:2])

Y_test = X_test_norm @ W

plt.figure()
#for i, name in zip( [0, 1, 2], t_names):
#    plt.scatter(Y_test[t_train == i, 0], Y_test[t_train == i, 1], alpha=.8, label=name)
for i, name in zip( [0, 1, 2], t_names):
    plt.scatter(Y_test[t_train == i, 0], Y_test[t_train == i, 1], alpha=.8, label=name)

# Fronteira de decisão e bgcolor
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
P = np.c_[xx.ravel(), xx.ravel(), yy.ravel(), yy.ravel()]
Z = P @ W
Z = np.where(Z >= 0, 1, 0)
Z = np.c_[Z,np.where(Z >= 0, 0, 1)]
Z = Z[:,1].reshape(xx.shape)
#plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes', norm=colors.Normalize(0., 1.), zorder=0)
#plt.contour(xx, yy, Z, [0], linewidths=2., colors='white')


plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.legend(loc='best', scatterpoints=1)



plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.savefig('figures/Fig4.3.4.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

##########################################################
### Opção 3 PRML
#model = FishersLinearDiscriminant()
#model.fit(X_train, t_train)
#y = model.classify(X_test)


###########################################################

### Opção 4 - Sklearn
#lda = LinearDiscriminantAnalysis()
#lda.fit(X_train, t_train)
#X_r2 = lda.transform(X_test)
#lda_pred = lda.predict(X_test)
#plt.figure()
#for i, name in zip( [0, 1, 2], t_names):
#    plt.scatter(X_r2[t == i, 0], X_r2[t == i, 1], alpha=.8, label=name)
#plt.legend(loc='best', shadow=False, scatterpoints=1)
