# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
iris = ds.load_iris()
X = iris.data
t = iris.target
names = iris.target_names
X1 = np.asarray(X[t==0,:])
X2 = np.asarray(X[t==1,:])
Xt = np.concatenate([X1[:25,:],X2[:25,:]])
tt = np.concatenate([t[:25], t[50:75]])
Xv = np.concatenate([X1[25:,:],X2[25:,:]])
m1 = np.mean(Xt[tt == 0,:], axis=0)
m2 = np.mean(Xt[tt == 1,:], axis=0)
S1 = np.zeros((4,4))
S2 = np.zeros((4,4)) 
for xi in Xt[tt== 0]: S1 += (xi.reshape(4,1)-m1).dot((xi.reshape(4,1)-m1).T)
for xi in Xt[tt == 1]: S2 += (xi.reshape(4,1)-m2).dot((xi.reshape(4,1)-m2).T)  
SW = S1 + S2
w = np.linalg.inv(SW) @ (m2-m1)
yt = Xt @ w
yv = Xv @ w

#plt.figure(figsize=(10,7))
#for i, nm in zip([0, 1], [names[0],names[1]]):
#    plt.scatter(Xv[tt == i,3], yv[tt == i], alpha=0.7, label=nm)
#plt.legend(loc='best', scatterpoints=1)
#plt.savefig('figures/Fig4.3.1.1.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


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
lda = LinearDiscriminantAnalysis()
lda.fit(Xt, tt)
lda_scores = lda.transform(Xv)
lda_pred = lda.predict(Xv)
#plt.figure()
#plt.plot(lda_scores, np.zeros(50), c='r') #lda_scores[t_train == i, 0])
#for i, name in zip([0, 1], [t_names[1],t_names[2]]):
#    # plt.scatter(lda_scores[t_train == i, 0], lda_scores[t_train == i, 0], alpha=.8, label=name)
#    plt.scatter(X_test[t_train == i,3], lda_scores[t_train == i, 0], alpha=.7)
#    #plt.plot(lda_scores[t_train == i, 0], np.zeros(25)) #lda_scores[t_train == i, 0])