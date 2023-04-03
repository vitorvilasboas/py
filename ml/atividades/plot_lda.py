"""
====================================================================
Linear and Quadratic Discriminant Analysis with covariance ellipsoid
====================================================================

This example plots the covariance ellipsoids of each class and
decision boundary learned by LDA and QDA. The ellipsoids display
the double standard deviation for each class. With LDA, the
standard deviation is the same for all the classes, while each
class has its own standard deviation with QDA.
"""
print(__doc__)

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.datasets as ds 

# #############################################################################
# Colormap
cmap = colors.LinearSegmentedColormap('red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

n, dim = 50, 2
np.random.seed(0)


iris = ds.load_iris()
X = iris.data  # features
t = iris.target
t_names = iris.target_names
X1 = np.asarray(X[t == 0,:])
X2 = np.asarray(X[t == 1,:])
X3 = np.asarray(X[t == 2,:])
#X = np.concatenate([X1,X2])
#y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

X = np.concatenate([X1[:25,:],X2[:25,:]])
y = np.concatenate([np.zeros(25, dtype=int), np.ones(25, dtype=int)])
Xv = np.concatenate([X1[25:,:],X2[25:,:]])

#features names = sepal length, sepal width, petal length, petal width 

#C = np.array([[0., -0.23], [0.83, .23]])
#X = np.r_[np.dot(np.random.randn(n, dim), C), np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
#y = np.hstack((np.zeros(n), np.ones(n)))


# Linear Discriminant Analysis
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
#xx, yy = np.meshgrid(X[y == 0,:], X[y == 1,:])
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
#plt.plot(xx,yy, marker='.', color='k', linestyle='none')
Z = lda.predict_proba(np.c_[xx.ravel(), xx.ravel(), yy.ravel(), yy.ravel()])
#Z = lda.predict(np.c_[xx.reshape(-1,2), yy.reshape(-1,2)])
#Z = lda.predict_proba(X)
Z = Z[:, 1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes', norm=colors.Normalize(0., 1.), zorder=0)
plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='best', scatterpoints=1)

## means
#plt.plot(lda.means_[0][0], lda.means_[0][1], '*', color='yellow', markersize=15, markeredgecolor='grey')
#plt.plot(lda.means_[1][0], lda.means_[1][1], '*', color='yellow', markersize=15, markeredgecolor='grey')
#
#v, w = linalg.eigh(lda.covariance_)
#u = w[0] / linalg.norm(w[0])
#angle = np.arctan(u[1] / u[0])
#angle = 180 * angle / np.pi  # convert to degrees
## filled Gaussian at 2 standard deviation
#ell = mpl.patches.Ellipse(lda.means_[0], 2 * v[0] ** 0.5, 2 * v[1] ** 0.5, 180 + angle, facecolor='red',
#                          edgecolor='black', linewidth=2)
#ell.set_clip_box(splot.bbox)
#ell.set_alpha(0.2)
#splot.add_artist(ell)
#
#ell = mpl.patches.Ellipse(lda.means_[1], 2 * v[0] ** 0.5, 2 * v[1] ** 0.5, 180 + angle, facecolor='red',
#                          edgecolor='black', linewidth=2)
#ell.set_clip_box(splot.bbox)
#ell.set_alpha(0.2)
#splot.add_artist(ell)
#
#splot.set_xticks(())
#splot.set_yticks(())
#
#
#plt.axis('tight')
#plt.tight_layout()
#plt.subplots_adjust(top=0.92)
#plt.show()
