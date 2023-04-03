# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:38:06 2019
@author: Vitor Vilas-Boas
Machine Learning - Activity 6 - Neural Networks e SVM

6. Considere o conjunto de dados “Handwritten Digits” descrito na página 677 do livro texto. 
Resolva o problema de classificação binário entre os dígitos 6 e 8, utilizando: 
6.1. Uma rede neural do tipo MLP. Discuta os resultados com base na determinação dos parâmetros da rede, 
como número de camadas ocultas, quantidade de épocas, taxa de aprendizagem, tempo de treinamento, funções de ativação; 
6.2. O algoritmo SVM. Discuta os resultados com base na função Kernel e do parâmetro C. 
Ver seção 7.1.1 do livro texto. 
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

digits = load_digits()
X1 = digits.data[np.where(digits.target==8)]
X2 = digits.data[np.where(digits.target==6)][:len(X1)]
X = np.vstack([X1,X2]) #~np.concatenate axis=0
t = np.hstack((np.ones(len(X1)), -np.ones(len(X2)))) #~np.concatenate axis=1

# XT, XV, tt, tv = train_test_split(X, t, test_size=0.5, random_state=42)
D = len(X[0])

folds = 10
kfold = StratifiedShuffleSplit(folds, test_size=0.2, random_state=42) # shuffle e random_state(semente geradora) garantem a aleatoriedade na escolha dos dados de treino e teste

eta = 0.1 # [10**(-2), 10**(-2), 10**(-2), 10**(-4), 10**(-5)]
epocas = 100 # n_iterações [10, 100, 1000, 10000]
M1 = M2 = np.random.randint(D+1,D*2) #np.linspace(D+1,D*3,10)
h_a = 'sigmoid' #sigmoid ou tanh

ACCs = []
MCs = []
for idx_treino, idx_teste in kfold.split(X, t):
    XT = X[idx_treino]
    XV = X[idx_teste]
    tt = t[idx_treino]
    tv = t[idx_teste]
    
    ##### TREINO #######
    N, D = XT.shape
    K = 1

    W1 = np.random.randn(M1,D+1);
    W2 = np.random.randn(M2,M1+1);
    W3 = np.random.randn(K,M2+1);
    
    #X = np.concatenate([np.ones((N,1)),X], axis=1)
    
    mse = []
    for n_iter in range(epocas):
        E = 0
        for n in range(N):
            ##### FORWARD ######
            x = XT[n,:]
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
            ##### BACKWARD ######
            tn = tt[n]
            deltaK = y-tn # Calculo do Erro >> [erro * sigma'(ak)]
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
    
    ###### VALIDAÇÃO #######
    N, D = XV.shape
    y = []
    for n in range(N):
        x = XV[n,:]
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
        y.append(W3.dot(zj2))
    
    #scores = np.where(np.asarray(np.ravel(y))>=0, 1, -1)    
    scores = np.asarray([1 if ydx >= 0 else -1 for ydx in y])
    
    acc = round(np.mean(scores == tv) * 100, 2)
    
    print(f'Acurácia: {acc}')
    
    MCs.append(confusion_matrix(tv, scores))
    ACCs.append(acc)
mc = np.mean(MCs, axis = 0)
acc_media = np.asarray(ACCs).mean()
print(f'Acc Média: {acc_media}')

#import matplotlib.pyplot as plt
#print(digits.data.shape)
#plt.gray() 
#plt.matshow(digits.images[0]) 
#plt.show() 