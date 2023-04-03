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
#from sklearn.metrics import confusion_matrix


digits = load_digits()
X1 = digits.data[np.where(digits.target==8)]
X2 = digits.data[np.where(digits.target==6)][:len(X1)] # necessário pq há mais reg 6 do que 8 (aki garante igualdade)
X = np.vstack([X1,X2]) #~np.concatenate axis=0
t = np.hstack((np.ones(len(X1)), -np.ones(len(X2)))) #~np.concatenate axis=1

# XT, XV, tt, tv = train_test_split(X, t, test_size=0.5, random_state=42)

### MEU SVM ####
def linear(x1, x2): return np.dot(x1, x2)
def poly(x, y, grau=2): return ((x.T @ y)+1)**grau
def sigmoid(x, y): return 1./(1 + np.exp(np.dot(x,y)))
def rbf(x, y, var=5.0): return np.exp(-linalg.norm(x-y)**2/(2*(var**2)))

folds = 10
kfold = StratifiedShuffleSplit(folds, test_size=0.2, random_state=42) # shuffle e random_state(semente geradora) garantem a aleatoriedade na escolha dos dados de treino e teste
ACCs = []
#MCs = []
FINAL = []

klabel = ['linear','poly','sigmoid','rbf']
kernel = [linear,poly,sigmoid,rbf]
C = np.linspace(1e-3,2,20)

kf = kernel[0]
#c = float(0.0001)

for kf,kl in zip(kernel, klabel):
    for c in C:

        for idx_treino, idx_teste in kfold.split(X, t):   
            XT = X[idx_treino]
            XV = X[idx_teste]
            tt = t[idx_treino]
            tv = t[idx_teste]
            
            ## TREINAMENTO
            N, D = XT.shape
            
            # Gram matrix
            K = np.zeros((N, N))
            for i in range(N):
                for j in range(N): K[i,j] = kf(XT[i], XT[j])
            
            P = cvxopt.matrix(np.outer(tt,tt)*K)
            q = cvxopt.matrix(-np.ones(N))
            A = cvxopt.matrix(tt, (1,N))
            bqp = cvxopt.matrix(0.0)
            G = cvxopt.matrix(np.vstack((np.diag(-np.ones(N)), np.identity(N))))
            h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N)*c)))
            
            qp = cvxopt.solvers.qp(P, q, G, h, A, bqp) # Resolve QuadProg
            a = np.ravel(qp['x']) # multiplicadores de Lagrange
            
            tol = 1e-10
            vets = a > tol
            ind = np.arange(len(a))[vets]
            a_vs = a[vets] # Seleciona vetores suporte a partir dos multiplicadores de lagrange
            XT_vs = XT[vets]
            y_vs = tt[vets]
            # print(f"{len(a_vs)} vetores suporte de um total de {N} exemplos")
            
            b = 0
            for n in range(len(a_vs)):
                b += y_vs[n]
                b -= np.sum(a_vs * y_vs * K[ind[n],vets]) # Calculo ponto interceptação b
            b /= len(a_vs)
            
            if kf == linear: # Calculo vetor pesos w
                w = np.zeros(D)
                for n in range(len(a_vs)): w += a_vs[n] * y_vs[n] * XT_vs[n] 
            
            ## VALIDAÇÃO
            if kf == linear: y = np.dot(XV, w) + b
            else:
                y = np.zeros(len(XV))
                for i in range(len(XV)):
                    aux = 0
                    for ai, yi, xi in zip(a_vs, y_vs, XT_vs): aux += ai * yi * kf(XV[i], xi)
                    y[i] = aux
                y = y + b
            
            scores = np.sign(y)
            acc = round(np.mean(scores == tv) * 100, 2)
            print(f'Acurácia: {acc}')
            
            #MCs.append(confusion_matrix(tv, scores))
            ACCs.append(acc)
        #mc = np.mean(MCs, axis = 0)
        acc_media = np.asarray(ACCs).mean()
        print(f'Acc Média: {acc_media}')
        
        FINAL.append((kl,c,acc_media))

df1 = np.asarray(FINAL)
df = pd.DataFrame(FINAL, columns=['kernel','C','acc']) 

df.to_csv('result_svm.csv')

plt.figure(figsize=(20,12))
plt.xticks(df.iloc[:,1])
plt.xlabel('C')
plt.ylabel('Acc Média')
plt.grid(axis='both')
for kl in klabel:
    kdata = df.loc[np.nonzero(df['kernel']==kl)]
    plt.plot(kdata.iloc[:,1], kdata.iloc[:,2], label=r'$K={}$'.format(kl), marker='o')
#plt.legend(loc='best')
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)  
plt.savefig('Fig6.2.png', format='png', dpi=300, transparent=True, bbox_inches='tight') 