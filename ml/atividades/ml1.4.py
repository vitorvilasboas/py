# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np
import matplotlib.pyplot as plt

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

x_test = np.linspace(0,1,100, endpoint=False)
noise = np.random.randn(len(x_test))
t_test = np.sin(2*np.pi*x_test) + 0.1 * noise 

# Cálculo do erro médio quadrático de previsão em relação ao alvo a partir de modelos com várias complexidades M=[0..9]. Busca-se minimizar esse erro
# Posteriormente normalizado pelo tamanho da amostra N=len(x), útil para comparação entre amostras de diferentes tamanhos.
Erms = []
ErmsV = []
for i in range(10):
    A_train = np.asarray([x_train**j for j in range(i+1)]).T
    A_test = np.asarray([x_test**j for j in range(i+1)]).T
    
    w = np.linalg.solve(A_train.T @ A_train, A_train.T @ t_train)  # inv(A.T * A) * A.T * t
    # w = np.linalg.inv(A_train.T.dot(A_train)).dot(A_train.T.dot(t_train)) # test one
    # w = np.linalg.inv(A_train.T @ A_train) @ (A_train.T @ t_train) # test two
    # w = np.linalg.pinv(A_train) @ t_train # test three
    
    y_train = A_train @ w 
    y_test = A_test @ w

    Erms.append(np.sqrt(np.mean(np.square(y_train - t_train)))) # ~ math.sqrt( (y_train-t_train).T.dot(y_train-t_train)/len(x_train) )
    ErmsV.append(np.sqrt(np.mean(np.square( y_test - t_test )))) # ~ math.sqrt( (y_test - t_test).T.dot(y_test - t_test)/len(x_test) )

plt.figure(figsize=(10,7))                 
plt.plot(Erms, 'o-', mfc="none", mec="b", ms=10, c="b", label="Treino")
plt.plot(ErmsV, 'o-', mfc="none", mec="r", ms=10, c="r", label="Teste")
plt.legend(loc="best")
plt.xlabel('M', fontsize=16)
plt.ylabel('$E_{RMS}$', fontsize=16)
plt.xticks(np.linspace(0,9,10))
plt.yticks(np.linspace(0,1,6))
plt.ylim(0,1)
#plt.savefig('figures/Fig1.4.1b_.png', format='png', dpi=300, transparent=True, bbox_inches='tight')  