# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:38:06 2019
@author: Vitor Vilas-Boas
Machine Learning - Activity 6 - Neural Networks e SVM

6. Considere o conjunto de dados “Handwritten Digits” descrito na página 677 do livro texto. 
Resolva o problema de classificação binário entre os dígitos 6 e 8, utilizando: 
6.1. Uma rede neural do tipo MLP. Discuta os resultados com base na determinação dos 
parâmetros da rede, como número de camadas ocultas, quantidade de épocas, taxa de 
aprendizagem, tempo de treinamento, funções de ativação; 
6.2. O algoritmo SVM. Discuta os resultados com base na função Kernel e do parâmetro C. 
Ver seção 7.1.1 do livro texto. 
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pylab as pl
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

digits = load_digits()
X1 = digits.data[np.where(digits.target==8)]
X2 = digits.data[np.where(digits.target==6)][:len(X1)] # necessário pq há mais reg 6 do que 8 (aki garante igualdade)
X = np.vstack([X1,X2]) #~np.concatenate axis=0
t = np.hstack((np.ones(len(X1)), -np.ones(len(X2)))) #~np.concatenate axis=1
# XT, XV, tt, tv = train_test_split(X, t, test_size=0.5, random_state=42)

folds = 10
kfold = StratifiedShuffleSplit(folds, test_size=0.2, random_state=42) # shuffle e random_state(semente geradora) garantem a aleatoriedade na escolha dos dados de treino e teste
ACCs = []
MCs = []

kernel = ['linear','poly','sigmoid','rbf']
C = np.linspace(1e-3,2,20)
#C = list([1e-9, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

cross_scores = []
mat_conf = []
FINAL = []

#kf = kernel[0]
#c = float(0.0001)

for kf in kernel:
    for c in C:

        clf = SVC(kernel=kf, C=c)
        for idx_treino, idx_teste in kfold.split(X, t):   
            XT = X[idx_treino]
            XV = X[idx_teste]
            tt = t[idx_treino]
            tv = t[idx_teste]
        
            clf.fit(XT, tt)
            scores = clf.predict(XV)
            acc = round(np.mean(scores == tv) * 100, 2)
            print(f'Acurácia: {acc}')
            
            mat_conf.append(confusion_matrix(tv, scores))
            ACCs.append(acc)
        mc = np.mean(mat_conf, axis = 0)
        acc_media = np.asarray(ACCs).mean()
        print(f'Acc Média: {acc_media}')
        FINAL.append((kf,c,acc_media))

df1 = np.asarray(FINAL)
df = pd.DataFrame(FINAL[:], columns=['kernel','C','acc'])

df.to_csv('result_svm_skl.csv')

plt.figure(figsize=(20,12))
plt.xticks(df.iloc[:,1])
plt.xlabel('C')
plt.ylabel('Acc Média')
plt.grid(axis='both')
for kf in kernel:
    kdata = df.loc[np.nonzero(df['kernel']==kf)]
    plt.plot(kdata.iloc[:,1], kdata.iloc[:,2], label=r'$K={}$'.format(kf), marker='o')
#plt.legend(loc='best')
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)  
plt.savefig('Fig6.2_skl.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

    
    