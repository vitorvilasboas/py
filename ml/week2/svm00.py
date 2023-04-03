import numpy as np
import matplotlib.pyplot as plt
import quadprog
import cvxopt
import qpsolvers
from scipy.linalg import eigh 
from scipy import optimize
import osqp
import scipy.sparse as spa

def quadprog1(P, q, G=None, h=None, A=None, b=None,
             initvals=None, verbose=True):
    l = -np.inf * np.ones(len(h))
    if A is not None:
        qp_A = spa.vstack([G, A]).tocsc()
        qp_l = np.hstack([l, b])
        qp_u = np.hstack([h, b])
    else:  # no equality constraint
        qp_A = G
        qp_l = l
        qp_u = h
    model = osqp.OSQP()
    model.setup(P=P, q=q,
                A=qp_A, l=qp_l, u=qp_u, verbose=verbose)
    if initvals is not None:
        model.warm_start(x=initvals)
    results = model.solve()
    return results.x, results.info.status


x = np.random.rand(4,100)

x[2:4,:] = x[2:4,:] + 5 * np.ones([2,100])

data = np.concatenate([x[0:2,:], x[2:4,:]], axis=-1)
#
#plt.figure(figsize=(10,7))
#plt.scatter(data[0,:],data[1,:], c='g', marker='o')
#plt.scatter(x[2,:],x[3,:], c='b', marker='^')

T = np.ones(200)
T[:100] = -1

D, N = data.shape

H1 = np.zeros([200,200])
for j in range(N):
    for i in range(N):
        H1[i,j] = ( (data[:,i].T @ data[:,j]) + 1 )**2
        H1[i,j] = T[i]*H1[i,j]*T[j]   
H1 = np.asarray(H1)

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




