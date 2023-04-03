import numpy as np
import matplotlib.pyplot as plt
import quadprog
import qpsolvers
from scipy.linalg import eigh 
from scipy import optimize
import osqp
import scipy.sparse as spa


Xa = np.random.randn(2,100)
Xb = np.random.randn(2,100) + 4 * np.ones([2,100])
data = np.concatenate([Xa, Xb], axis=-1)


#
#x = np.random.randn(4,100)
#x[2:4,:] = x[2:4,:] + 4 * np.ones([2,100])
#data1 = np.concatenate([x[0:2,:], x[2:4,:]], axis=-1)
#
#plt.figure(figsize=(10,7))
#plt.scatter(data1[0,:], data1[1,:], c='g', marker='o')
#plt.scatter(x[2,:],x[3,:], c='b', marker='^')
#
#plt.figure(figsize=(10,7))
#plt.scatter(data[0,:], data[1,:], c='g', marker='o')
#plt.scatter(Xa,Xb, c='b', marker='^')

T = np.ones(200)
T[:100] = -1






D, N = data.shape

kernel = lambda v1,v2: (v1.T @ v2 + 1)**2

H = np.zeros([N,N])
for j in range(N):
    for i in range(N):
        H1[i,j] = kernel(data[:,i], data[:,j])
        H1[i,j] = T[i]*H1[i,j]*T[j]   
H = np.asarray(H1)

#H1 = H1 * np.identity(N) * 1e-10

K = ( data.T @ data + 1 ) ** 2 # alternativa

#TT = []
#for t1 in T:
#    TT.append([ t1 * t2 for t2 in T])
###TT = np.asarray(TT)
##H = TT * K

[D1, W1] = eigh(H1)
#[D2, W2] = eigh(H)

f = -np.ones(200)
Aeq = T
Beq = np.zeros([1,1])
lb = np.zeros([1,200])
ub = np.inf * np.ones(N)

a, status = quadprog1(H1, f, [], [], Aeq, Beq)

#a = quadprog.solve_qp(H1,f)#,[ ],np.empty,Aeq,Beq)
#a = quadprog.solve_qp(H,f,[],[],Aeq,Beq,lb,ub,[])
#a = qpsolvers.solve_qp(H1,f,[],[],Aeq,Beq)
#a = cvxopt.solvers.qp(H1,f,[],[],Aeq,Beq)

#a = optimize.minimize(H1, f)

#plt.plot(a)

#indices = np.empty([1,N])
#tol = 1e-6
#indices = indices[a > tol]




