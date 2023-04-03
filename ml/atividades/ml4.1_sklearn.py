from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.datasets as ds 

cmap = colors.LinearSegmentedColormap('red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

np.random.seed(0)

iris = ds.load_iris()
X = iris.data  
t = iris.target
t_names = iris.target_names
X1 = np.asarray(X[t == 0,:])
X2 = np.asarray(X[t == 1,:])
X3 = np.asarray(X[t == 2,:])

X = np.concatenate([X1[:25,:],X2[:25,:]])
y = np.concatenate([np.zeros(25, dtype=int), np.ones(25, dtype=int)])
Xv = np.concatenate([X1[25:,:],X2[25:,:]])

#features names = sepal length, sepal width, petal length, petal width 
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(X, y).predict(Xv)

splot = plt.subplot(1, 1, 1)
tp = (y == y_pred)  # verdadeiros positivos
tp0, tp1 = tp[y == 0], tp[y == 1]
X0, X1 = Xv[y == 0], Xv[y == 1]
X0_tp, X0_fp = X0[tp0], X0[~tp0]
X1_tp, X1_fp = X1[tp1], X1[~tp1]

plt.figure(figsize=(10, 8), facecolor='white')
plt.scatter(X0_tp[:, 0], X0_tp[:, 2], marker='.', color='red', alpha=0.7, label=t_names[0])
plt.scatter(X0_fp[:, 0], X0_fp[:, 2], marker='x', s=20, color='#990000')  # dark red
plt.scatter(X1_tp[:, 0], X1_tp[:, 2], marker='.', color='blue', alpha=0.7, label=t_names[1])
plt.scatter(X1_fp[:, 0], X1_fp[:, 2], marker='x', s=20, color='#000099')  # dark blue

nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
Z = lda.predict_proba(np.c_[xx.ravel(), xx.ravel(), yy.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes', norm=colors.Normalize(0., 1.), zorder=0)
plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='best', scatterpoints=1)