# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# F(x): Função alvo - Para fins didaticos é conhecida, mas normalmente é desconhecida! é o que deseja-se aprender.
x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

## Dataset Treinamento com rótulos=alvo=target -> mapeamento de x para y (x --> y)
x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

## Dataset Teste (N maior - mais dados)
x_test = np.linspace(0.01,0.99,1000, endpoint=False)
noise = np.random.randn(len(x_test))
# noise = np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
t_test = np.sin(2*np.pi*x_test) + 0.1 * noise # esse mesmo processo gerou a amostra de treinamento - aqui, somente para ilustração - em geral não se conhece os alvos em dados de teste

## Modelo de aprendizagem - Função Hipótese (g(x)) que quer aproximar f(x)
M = [0,1,3,9] # complexidade do modelo - n° de parametros livres - que na forma polinomial define o grau progressivo do polinomio
# ... para representar as M+1 características dos dados de entrada

plt.subplots(1, figsize=(10,6))
for j, grau in enumerate(M):
    # A é M+1 para considerar já com o bias = x0 = 1 representando o limiar que foi convertido para peso w0 - A = [ [x⁰],[x¹],[x²],...,[x^M] ].T
    A_train = np.asarray([x_train**i for i in range(grau+1)]).T
    A_test = np.asarray([x_test**i for i in range(grau+1)]).T
    
    # conjunto de parametros livres que minimizam o erro entre y e t
    w = np.linalg.pinv(A_train) @ t_train # inv(A.T * A) * A.T * t 
    
    # Aplicando modelo aos próprios dados que o treinaram
    y_train = A_train @ w 
    
    # Aplicando w aos dados de teste (desconhecidos por w) "generalização"
    y_test = A_test @ w # y_test é a saida do modelo = f(x,w) = A*w
    
    Err = np.mean(np.square(A_train @ w - t_train)) # Err = var
    y_std = np.sqrt(Err) + np.zeros_like(y_test)
    
    
#    Usando scikit-learn
#    polynomial_features= PolynomialFeatures(degree=grau)
#    A_train = polynomial_features.fit_transform(x_train[:, np.newaxis])
#    A_test = polynomial_features.fit_transform(x_test[:, np.newaxis])
#    
#    model = LinearRegression()
#    model.fit(A_train, t_train)
#    y_train = model.predict(A_train)
#    y_test = model.predict(A_test)
    
    plt.subplot(2,2,j+1)
    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
    plt.plot(x_correct, t_correct, c="g", alpha=0.6, label="$\sin(2\pi x)$")
    plt.plot(x_test, y_test, c="r", label="$y(x,w)$") #fitting
    plt.ylabel('$t$', fontsize=13, rotation = 360)
    plt.xlabel('$\mathit{x}$', fontsize=13,labelpad=-15)
    plt.xticks([0,1])
    plt.yticks([-1,0,1])
    plt.ylim(-1.5,1.5)
    plt.annotate(('M={}'.format(grau)), xy=(0.8, 1))
    
plt.legend(bbox_to_anchor=(1.05, 2.2), loc=2, borderaxespad=0.)
#plt.legend(loc='lower left')
plt.savefig('figures/Fig1.3.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
