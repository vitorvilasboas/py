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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from time import time

digits = load_digits()
X1 = digits.data[np.where(digits.target==8)]
X2 = digits.data[np.where(digits.target==6)][:len(X1)]
X = np.vstack([X1,X2]) #~np.concatenate axis=0
t = np.hstack((np.ones(len(X1)), -np.ones(len(X2)))) #~np.concatenate axis=1

# XT, XV, tt, tv = train_test_split(X, t, test_size=0.5, random_state=42)
D = len(X[0])

folds = 10
kfold = StratifiedShuffleSplit(folds, test_size=0.2, random_state=42) # shuffle e random_state(semente geradora) garantem a aleatoriedade na escolha dos dados de treino e teste

FINAL_1h = []

n_neuronios = np.linspace(D+1,D*5,10)  
epochs = [10, 100, 1000] # n_iterações
learn = [10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1)] #0.01 
fact = ['tanh'] #['sigmoid','tanh']

ACCs = []
MCs = []
COST = []

K = 1

M = 2 * D 
#eta = 0.01

#for M in n_neuronios:
W1 = np.random.randn(int(M),D+1)
W2 = np.random.randn(K,int(M)+1)

for epocas in epochs:

    for eta in learn:
    
            for h_a in fact:
                
                for idx_treino, idx_teste in kfold.split(X, t):
                    XT = X[idx_treino]
                    XV = X[idx_teste]
                    tt = t[idx_treino]
                    tv = t[idx_teste]
                    
                    ##### TREINO #######
                    N, D = XT.shape
                    
                    #X = np.concatenate([np.ones((N,1)),X], axis=1)
                    mse = []
                    start = time()
                    for n_iter in range(epocas):
                        E = 0
                        for n in range(N):
                            ##### FORWARD ######
                            x = XT[n,:]
                            x = np.concatenate([np.ones(1),x]) #add bias
                            #% Camada Oculta
                            aj = W1.dot(x)
                            #print(aj)
                            if h_a == 'sigmoid': ha = 1./(1 + np.exp(-aj))
                            elif h_a == 'tanh':  ha = np.tanh(aj) #(np.exp(aj) - np.exp(-aj))/(np.exp(aj) + np.exp(-aj))
                            zj = np.array(np.concatenate([np.ones(1), np.asarray(ha)]))
                            #% Saída
                            y = W2.dot(zj)
                            ##### BACKWARD ######
                            tn = tt[n]
                            deltaK = y-tn # Calculo do Erro >> [erro * sigma'(ak)]
                            #% Propagação do erro na camada oculta
                            if h_a == 'sigmoid': dz = zj * (1-zj)
                            elif h_a == 'tanh':  dz = 1 - zj**2
                            deltaJ = dz * (W2.T @ deltaK)
                            deltaJ = deltaJ[1:]
                            #% Ajuste formato vetores/matrizes
                            x = x.reshape(-1,1)
                            zj = zj.reshape(-1,1)
                            deltaK = deltaK.reshape(-1,1)
                            deltaJ = deltaJ.reshape(-1,1)     
                            #% Atualiza pesos
                            W2 = W2 - eta * deltaK.dot(zj.T)
                            W1 = W1 - eta * deltaJ.dot(x.T)        
                            # Soma dos erros entre y e t em cada exemplo x
                            E += deltaK.T.dot(deltaK)
                        mse.append(E/N) # Calcula e armazena Erro Quadrático Médio
                    cost = time() - start
                    mse = np.ravel(mse)
                    
                    ###### VALIDAÇÃO #######
                    N, D = XV.shape
                    y = []
                    for n in range(N):
                        x = XV[n,:]
                        x = np.concatenate([np.ones(1),x]) #add bias
                        #% Camada Oculta
                        aj = W1.dot(x)
                        if h_a == 'sigmoid': ha = 1./(1 + np.exp(-aj))
                        elif h_a == 'tanh':  ha = np.tanh(aj) #(np.exp(aj) - np.exp(-aj))/(np.exp(aj) + np.exp(-aj))
                        zj = np.array(np.concatenate([np.ones(1), np.asarray(ha)]))
                        #% Saída
                        y.append(W2.dot(zj))
                    
                    #scores = np.where(np.asarray(np.ravel(y))>=0, 1, -1)    
                    scores = np.asarray([1 if ydx >= 0 else -1 for ydx in y])
                    
                    acc = round(np.mean(scores == tv) * 100, 2)
                    
                    #print(f'Acurácia: {acc}')
                    
                    MCs.append(confusion_matrix(tv, scores))
                    ACCs.append(acc)
                    COST.append(cost)
                mc = np.mean(MCs, axis = 0)
                acc_media = np.asarray(ACCs).mean()
                custo_medio = np.asarray(COST).mean()
                print(f'Acc Média: {acc_media}')
                
                FINAL_1h.append((h_a,1,M,epocas,eta,acc_media,custo_medio))

df_1h = pd.DataFrame(FINAL_1h, columns=['h_a','n_hidden','M','epochs','eta','acc_mean','cost_mean'])

plt.figure(figsize=(10,7))
plt.xticks(df_1h.iloc[:,3])
plt.xlabel('Iterações')
plt.ylabel('Acc Média')
for eta in learn:
    eta_data = df_1h.loc[np.nonzero(df_1h['eta']==eta)]
    plt.plot(eta_data.iloc[:,3], eta_data.iloc[:,5], label=r'$eta={}$'.format(eta))
plt.legend(loc='best')
#plt.savefig('Fig6.1.png', format='png', dpi=300, transparent=True, bbox_inches='tight') 


print(df_1h)

#df_1h.to_csv()

#import matplotlib.pyplot as plt
#print(digits.data.shape)
#plt.gray() 
#plt.matshow(digits.images[0]) 
#plt.show() 