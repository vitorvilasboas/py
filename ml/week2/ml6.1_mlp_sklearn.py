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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score
from time import time
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

digits = load_digits()
X1 = digits.data[np.where(digits.target==8)]
X2 = digits.data[np.where(digits.target==6)][:len(X1)]
X = np.vstack([X1,X2]) #~np.concatenate axis=0
t = np.hstack((np.ones(len(X1)), -np.ones(len(X2)))) #~np.concatenate axis=1

D = len(X[0])

folds = 10
kfold = StratifiedShuffleSplit(folds, test_size=0.2, random_state=42) # shuffle e random_state(semente geradora) garantem a aleatoriedade na escolha dos dados de treino e teste
# XT, XV, tt, tv = train_test_split(X, t, test_size=0.5, random_state=42)
hidden = [1,2,3]
#M1 = M2 = np.random.randint(D+1,D*5) #np.linspace(D+1,D*5,10)
M = 2 * D
# epocas = 1000 # n_iterações 
# eta = 0.01 # [10**(-2), 10**(-2), 10**(-2), 10**(-4), 10**(-5)]
# fact = ['logistic','tanh'] #sigmoid ou tanh

n_neuronios = np.linspace(D+1,D*5,10)  
epochs = [10, 100, 1000] # n_iterações
learn = [10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1)] #0.01 
fact = ['tanh'] #['sigmoid','tanh']


FINAL = []

#for nh in hidden:
nh = 1
for eta in learn:
    for epocas in epochs:
        for h_a in fact:  
            ACCs = []
            COST = []
            MCs = []
            for idx_treino, idx_teste in kfold.split(X, t):
                XT = X[idx_treino]
                XV = X[idx_teste]
                tt = t[idx_treino]
                tv = t[idx_teste]
                ##### TREINO #######
                mse = []
                clf = MLPClassifier(learning_rate_init = eta, max_iter=epocas,
                                    hidden_layer_sizes=(nh, M), activation=h_a)
                start = time()
                clf.fit(XT, tt)
                cost = time() - start
                mse = np.ravel(clf.loss_curve_)
                
                ###### VALIDAÇÃO #######
                scores = clf.predict(XV)
                acc = round(accuracy_score(tv, scores) * 100, 2)
                
                MCs.append(confusion_matrix(tv, scores))
                ACCs.append(acc)
                COST.append(cost)
            mc = np.mean(MCs, axis = 0)
            acc_media = np.asarray(ACCs).mean()
            custo_medio = np.asarray(COST).mean()
            FINAL.append((h_a,nh,M,epocas,eta,acc_media,custo_medio))

df = pd.DataFrame(FINAL, columns=['h_a','n_hidden','M','epochs','eta','acc_mean','cost_mean'])

plt.figure(figsize=(10,7))
plt.xticks(df.iloc[:,3])
plt.xlabel('Iterações')
plt.ylabel('Acc Média')
for eta in learn:
    eta_data = df.loc[np.nonzero(df['eta']==eta)]
    plt.plot(eta_data.iloc[:,3], eta_data.iloc[:,5], label=r'$eta={}$'.format(eta))
plt.legend(loc='best')
plt.savefig('Fig6.1.png', format='png', dpi=300, transparent=True, bbox_inches='tight') 


print(df)