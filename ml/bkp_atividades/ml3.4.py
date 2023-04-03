# -*- coding: utf-8 -*-

def update_w(W, x_n, y_n, phi, eta=0.01):
    phi_n = matrix(sapply(phi, function(base) base(x_n)), ncol=1) # make it a vector
    W + eta * phi_n %*% (y_n - t(W)%*%phi_n)
  

X = [1,2,3,5,7]
Y = [3,5,6,12,21]
phi = c(one, id) # basis for linear regression

eta = 0.015 # choosing value for eta is tricky...
convt=1e-3

W = rnorm(len(phi),0,0.1) # initialization to small random values
for i in range(1,length(X)): # batch update
    W = update_w(W, X[i], Y[i], phi, eta)



plot(X,Y,pch=19)
abline(W, col="red")