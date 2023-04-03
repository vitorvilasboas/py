# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

x_test = np.linspace(0.01,0.99,1000, endpoint=False)
noise = np.random.randn(len(x_test))
# noise = np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
t_test = np.sin(2*np.pi*x_test) + 0.1 * noise 

Erms = []
ErmsV = []

# Cálculo do erro de previsão em relação ao alvo - Dados de treinamento - Busca-se minimizar esse erro
# Calculando o erro a partir de modelos com várias complexidades M=[0 até 9]
# erro médio quadrático - é normalizado pelo tamanho da amostra N=len(x) (para comparar erro em amostras de vários tamanhos)
for i in range(10):
    A_train = np.asarray([x_train**j for j in range(i+1)]).T
    A_test = np.asarray([x_test**j for j in range(i+1)]).T
    
    w = np.linalg.pinv(A_train) @ t_train # inv(A.T * A) * A.T * t 
    y_train = A_train @ w 
    y_test = A_test @ w
    
    #Err = np.mean(np.square(A_train @ w - t_train)) # Err = var
    #y_std = np.sqrt(Err) + np.zeros_like(y_test)

    Erms.append(np.sqrt(np.mean(np.square(y_train - t_train)))) # ~ math.sqrt( (y_train-t_train).T.dot(y_train-t_train)/len(x_train) )
    ErmsV.append(np.sqrt(np.mean(np.square( y_test - t_test )))) # ~ math.sqrt( (y_test - t_test).T.dot(y_test - t_test)/len(x_test) )
    
#    Usando scikit-aulas
#    polynomial_features= PolynomialFeatures(degree=i)
#    A_train = polynomial_features.fit_transform(x_train[:, np.newaxis])
#    A_test = polynomial_features.fit_transform(x_test[:, np.newaxis])
#    
#    model = LinearRegression()
#    model.fit(A_train, t_train)
#    y_train_ = model.predict(A_train)
#    y_test = model.predict(A_test)
#    
#    rmse = np.sqrt(mean_squared_error(t_test,y_test))
#    r2 = r2_score(t_test,y_test)

plt.figure(figsize=(10,7))                 
plt.plot(Erms, 'o-', mfc="none", mec="b", ms=10, c="b", label="Training")
plt.plot(ErmsV, 'o-', mfc="none", mec="r", ms=10, c="r", label="Test")
plt.legend(loc="best")
plt.xlabel('M', fontsize=16)
plt.ylabel('$E_{RMS}$', fontsize=16)
plt.xticks(np.linspace(0,9,10))

#plt.yticks(np.linspace(0,1,6))
#plt.ylim(0,1)

plt.savefig('figures/Fig1.4.1.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

    

    