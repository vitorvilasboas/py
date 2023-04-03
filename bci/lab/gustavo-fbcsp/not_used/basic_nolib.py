# Script simples com CSP+LDA sem MNE/SKLearn
# 1 subject, 1 combinacao de classes

from numpy import cov, diag, dot, log, mean, trace, zeros
from scipy.linalg import eig, norm, pinv
from load_data import load_data

SUBJECT = 1
classes = [1,4]

X = load_data(SUBJECT, classes)

X1T = X[0][0]
X2T = X[0][1]
X1V = X[1][0]
X2V = X[1][1]

## Extract means

for i in range(72):
    for j in range(22):
        X1T[i,j,:] = X1T[i,j,:] - mean(X1T[i,j,:])
        X2T[i,j,:] = X2T[i,j,:] - mean(X2T[i,j,:])


## Make covariance matrices

S1T = zeros((22, 22))
S2T = zeros((22, 22))
for i in range(72):
   S1T = S1T + dot(X1T[:,:,i].T, X1T[:,:,i])
   S2T = S2T + dot(X2T[:,:,i].T, X2T[:,:,i])


## CSP

[_, W] = eig(S1T, S1T + S2T)

for i in range(72):
    X1T[:,:,i] = dot(X1T[:,:,i], W.T)
    X2T[:,:,i] = dot(X2T[:,:,i], W.T)
    X1V[:,:,i] = dot(X1V[:,:,i], W.T)
    X2V[:,:,i] = dot(X2V[:,:,i], W.T)


## Log-variance

X1T_VAR = zeros((22,72))
X2T_VAR = zeros((22,72))
X1V_VAR = zeros((22,72))
X2V_VAR = zeros((22,72))
for i in range(72):
   X1T_VAR[:,i] = log(diag(dot(X1T[:,:,i].T, X1T[:,:,i])))
   X2T_VAR[:,i] = log(diag(dot(X2T[:,:,i].T, X2T[:,:,i])))
   X1V_VAR[:,i] = log(diag(dot(X1V[:,:,i].T, X1V[:,:,i])))
   X2V_VAR[:,i] = log(diag(dot(X2V[:,:,i].T, X2V[:,:,i])))


## LDA

m1 = mean(X1T_VAR, 1)
m2 = mean(X2T_VAR, 1)

S1 = cov(X1T_VAR)
S1 = S1 / trace(S1)
S2 = cov(X1V_VAR)
S2 = S2 / trace(S2)
Sw = S1 + S2

B = dot(pinv(Sw), (m1-m2));
b = dot(B.T, m1+m2) / 2;

## Verification on test Data

acc1 = mean(dot(X1V_VAR.T, B) >= b)
acc2 = 1-mean(dot(X2V_VAR.T, B) >= b)
acc_test = (acc1 + acc2) / 2

print('Test accuracy: ' + str(acc_test*100))
