# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:38:06 2019
@author: Vitor Vilas-Boas
Machine Learning - Activity 6 - Neural Networks

6. Considere o conjunto de dados “Handwritten Digits” descrito na página 677 do livro texto. 
Resolva o problema de classificação binário entre os dígitos 6 e 8, utilizando: 
6.1. Uma rede neural do tipo MLP. Discuta os resultados com base na determinação dos 
parâmetros da rede, como número de camadas ocultas, quantidade de épocas, taxa de 
aprendizagem, tempo de treinamento, funções de ativação; 
6.2. O algoritmo SVM. Discuta os resultados com base na função Kernel e do parâmetro C. 
Ver seção 7.1.1 do livro texto. 
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold


digits = load_digits()
X6 = digits.data[np.where(digits.target==6)]
X8 = digits.data[np.where(digits.target==8)]
X = np.concatenate([X6[:len(X8)],X8])
T = np.concatenate([-np.ones(int(len(X)/2), dtype=int), np.ones(int(len(X)/2), dtype=int)])

###### 6.1. NN ############
N, D = X.shape
K = 1
epocas = 2 # n_iterações
eta = 0.1
h_a = 'sigmoid' #sigmoid or tanh
M1 = M2 = np.random.randint(D+1,D*2)
W1 = np.random.randn(M1,D+1);
W2 = np.random.randn(M2,M1+1);
W3 = np.random.randn(K,M2+1);

#X = np.concatenate([np.ones((N,1)),X], axis=1)
mse = []

X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.33, random_state=42)

for n_iter in range(epocas):
    Y = []
    E = 0
    for n in range(N):
        ##### FORWARD ######
        x = X[n,:]
        x = np.concatenate([np.ones(1),x]) #add bias
        #% 1a Camada Oculta
        aj1 = W1.dot(x)
        if h_a == 'sigmoid': ha1 = 1./(1 + np.exp(-aj1))
        elif h_a == 'tanh':  ha1 = (np.exp(aj1) - np.exp(-aj1))/(np.exp(aj1) + np.exp(-aj1))
        zj1 = np.array(np.concatenate([np.ones(1), np.asarray(ha1)]))
        #% 2a Camada Oculta
        aj2 = W2.dot(zj1)
        if h_a == 'sigmoid': ha2 = 1./(1 + np.exp(-aj2))
        elif h_a == 'tanh':  ha2 = (np.exp(aj2) - np.exp(-aj2))/(np.exp(aj2) + np.exp(-aj2))
        zj2 = np.array(np.concatenate([np.ones(1), np.asarray(ha2)]))
        #% Saída
        y = W3.dot(zj2)
        Y.append(y)
        ##### BACKWARD ######
        t = T[n]
        deltaK = y-t # Calculo do Erro >> [erro * sigma'(ak)]
        #% Propagação do erro na 2a camada oculta
        if h_a == 'sigmoid': dz2 = zj2 * (1-zj2)
        elif h_a == 'tanh':  dz2 = 1 - zj2**2
        deltaJ2 = dz2 * (W3.T @ deltaK)
        deltaJ2 = deltaJ2[1:]
        #% Propagação do erro na 1a camada oculta
        if h_a == 'sigmoid': dz1 = zj1 * (1-zj1)
        elif h_a == 'tanh':  dz1 = 1 - zj1**2
        deltaJ1 = dz1 * (W2.T @ deltaJ2)
        deltaJ1 = deltaJ1[1:]
        #% Ajuste formato vetores/matrizes
        x = x.reshape(-1,1)
        zj2 = zj2.reshape(-1,1)
        zj1 = zj1.reshape(-1,1)
        deltaK = deltaK.reshape(-1,1)
        deltaJ2 = deltaJ2.reshape(-1,1)
        deltaJ1 = deltaJ1.reshape(-1,1)     
        #% Atualiza pesos
        W3 = W3 - eta * deltaK.dot(zj2.T)
        W2 = W2 - eta * deltaJ2.dot(zj1.T)
        W1 = W1 - eta * deltaJ1.dot(x.T)        
        # Soma dos erros entre y e t em cada exemplo x
        E += deltaK.T.dot(deltaK)
    mse.append(E/N) # Calcula e armazena Erro Quadrático Médio
mse = np.ravel(mse)

scores = np.where(np.asarray(Y)>=0, 1, -1)

scores2 = np.asarray([1 if ydx >= 0 else -1 for ydx in Y])

acc = round(np.mean(scores2 == T) * 100, 2)

print(f'Acurácia: {acc}')

###### 6.2. SVM ############

#from sklearn import svm
#clf = svm.SVC(gamma=0.001, C=100)


#import matplotlib.pyplot as plt
#print(digits.data.shape)
#plt.gray() 
#plt.matshow(digits.images[0]) 
#plt.show() 


