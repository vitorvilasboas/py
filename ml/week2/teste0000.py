# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pylab as pl
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X1 = digits.data[np.where(digits.target==8)]
X2 = digits.data[np.where(digits.target==6)][:len(X1)] # necessário pq há mais reg 6 do que 8 (aki garante igualdade)
X = np.vstack([X1,X2]) #~np.concatenate axis=0
y = np.hstack((np.ones(len(X1)), -np.ones(len(X2)))) #~np.concatenate axis=1
XT, XV, yt, yv = train_test_split(X, y, test_size=0.3, random_state=42)

def klinear(x1, x2): return np.dot(x1, x2)
def kpoly(x, y, grau=3): return (np.dot(x, y) + 1)**grau
def ksigmoid(x): return 1./(1 + np.exp(x))
def kgauss(x, y, var=5.0): return np.exp(-linalg.norm(x-y)**2/(2*(var**2)))

kernel = klinear
C = float(2) #None

## TREINAMENTO
N, D = XT.shape

# Gram matrix
K = np.zeros((N, N))
for i in range(N):
    for j in range(N): 
        K[i,j] = kernel(XT[i], XT[j])

P = cvxopt.matrix(np.outer(yt,yt) * K)
q = cvxopt.matrix(np.ones(N) * -1)
A = cvxopt.matrix(yt, (1,N))
bqp = cvxopt.matrix(0.0)

if C is None:
    G = cvxopt.matrix(np.diag(-np.ones(N)))
    h = cvxopt.matrix(np.zeros(N))
else:
    G = cvxopt.matrix(np.vstack((np.diag(-np.ones(N)), np.identity(N))))
    h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N)*C)))

solution = cvxopt.solvers.qp(P, q, G, h, A, bqp) # Resolve QuadProg
a = np.ravel(solution['x']) # multiplicadores de Lagrange

tol = 1e-5
vets = a > tol
ind = np.arange(len(a))[vets]
a_vs = a[vets] # Seleciona vetores suporte a partir dos multiplicadores de lagrange
XT_vs = XT[vets]
y_vs = yt[vets]
print(f"{len(a_vs)} vetores suporte de um total de {N} exemplos")

b = 0
for n in range(len(a_vs)):
    b += y_vs[n]
    b -= np.sum(a_vs * y_vs * K[ind[n],vets]) # Calculo ponto interceptação b
b /= len(a_vs)

if kernel == klinear: # Calculo vetor pesos w
    w = np.zeros(D)
    for n in range(len(a_vs)): w += a_vs[n] * y_vs[n] * XT_vs[n] 

## VALIDAÇÃO
if kernel == klinear: scores = np.dot(XV, w) + b
else:
    scores0 = np.zeros(len(XV))
    for i in range(len(XV)):
        aux = 0
        for ai, yi, xi in zip(a_vs, y_vs, XT_vs): aux += ai * yi * kernel(XV[i], xi)
        scores0[i] = aux
    scores = scores0 + b

acc = round(np.mean(np.sign(scores) == yv) * 100, 2)
print(f"Acc: {acc}")