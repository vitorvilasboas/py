"""
===================================================================
Comparação da projeção 2D do LDA e do PCA do conjunto de dados Iris
===================================================================

O conjunto de dados Iris representa 3 tipos de flores Iris (Setosa, Versicolour 
e Virginica) com 4 atributos: comprimento da sépala, largura da sépala, comprimento da pétala 
e largura da pétala.

A Análise de Componente Principal (PCA) aplicada a esses dados identifica a 
combinação de atributos (componentes principais ou direções no 
espaço de caracteríticas) que respondem pela maior variação nos dados. Aqui, 
plotamos as diferentes amostras nos 2 primeiros componentes principais.

A Análise Discriminante Linear (LDA) tenta identificar atributos que 
representam a maior variação *entre as classes*. Em particular, 
o LDA, ao contrário do PCA, é um método supervisionado, usando rótulos de classe conhecidos.

"""
print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Porcentagem de variação explicada para cada componente
print('Taxa de variação explicada (Dois primeiros componentes): %s' % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 1

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
