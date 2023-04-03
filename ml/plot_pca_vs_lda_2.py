import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
X = iris.data 
t = iris.target
t_names = iris.target_names

lda = LinearDiscriminantAnalysis(n_components=2)
w = lda.fit(X, t)

X_r2 = lda.transform(X)


colors = ['navy', 'turquoise', 'darkorange']
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], t_names):
    plt.scatter(X_r2[t == i, 0], X_r2[t == i, 1], alpha=.8, color=color, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')



