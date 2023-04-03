# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np
import sklearn.datasets as ds
import cvxopt
import quadprog
from scipy.linalg import eigh
import math

iris = ds.load_iris()
X = iris.data
t = iris.target

X = np.concatenate([ np.asarray(X[t == 1,:]), np.asarray(X[t == 2,:]) ])
X = X[48:52]

Nc = math.floor(len(X)/2)
t = np.asarray( np.concatenate([ np.ones(Nc, dtype=int), -np.ones(Nc, dtype=int)]) )

teta = 1
beta = 1
grau = 2

K1 = []
for x in X:
    k = []
    for z in X: k.append((x.T @ z + 1) ** grau)
    K1.append(k)    
K1 = np.asarray(K1)

K2 = ( X @ X.T + teta ) ** grau # alternativa

T = []
for t1 in t:
    T.append([ t1 * t2 for t2 in t])
T = np.asarray(T)

#T = np.ones([len(X),len(X)])

# Ta = t.T @ t # alternativa

H = T * K

f = -np.ones(len(X))

a = quadprog.solve_qp(H,f)

# aa = quadprog(H,f,A,b,Aeq,beq,LB,UB,Xo)
#f = f.reshape(-1,1)
#aa = cvxopt.solvers.qp(H,f,[ ],[ ],t,0,0,np.inf,[ ])



