from numpy import *
from scipy.linalg import eig, norm, pinv


from load_data import load_data

PARES = 11
    
def csp_lda(SUBJECT, classes):

    X = load_data(SUBJECT, classes)

    X1T = X[0][0]
    X2T = X[0][1]
    X1V = X[1][0]
    X2V = X[1][1]


    ## Extract means

    for i in range(22):
        for j in range(72):
            X1T[i,:,j] = X1T[i,:,j] - mean(X1T[i,:,j])
            X2T[i,:,j] = X2T[i,:,j] - mean(X2T[i,:,j])


    ## Make covariance matrices

    S1T = zeros((22, 22))
    S2T = zeros((22, 22))
    for i in range(72):
    
        F = X1T[:,:,i]
        S1T = S1T + dot(F.T, F)
        
        F = X2T[:,:,i]
        S2T = S2T + dot(F.T, F)


    ## CSP

    [D, W] = eig(S1T, S1T + S2T)
    idx = abs(D).argsort()
    idx = idx[range(PARES) + list(arange(22-PARES,22))]
    D = D[idx]
    W = W[:,idx]
    
    X1T_CSP = zeros((PARES*2, 1000, 72))
    X2T_CSP = zeros((PARES*2, 1000, 72))
    X1V_CSP = zeros((PARES*2, 1000, 72))
    X2V_CSP = zeros((PARES*2, 1000, 72))
    for i in range(72):
        print W.shape
        print X1T[:,:,i].shape
        x = dot(W.T, X1T[:,:,i])
        print x.shape
        X1T_CSP[:,:,i] = dot(W.T, X1T[:,:,i])
        X2T_CSP[:,:,i] = dot(W.T, X2T[:,:,i])
        X1V_CSP[:,:,i] = dot(W.T, X1V[:,:,i])
        X2V_CSP[:,:,i] = dot(W.T, X2V[:,:,i])


    ## Log-variance

    X1T_VAR = zeros((PARES*2, 72))
    X2T_VAR = zeros((PARES*2, 72))
    X1V_VAR = zeros((PARES*2, 72))
    X2V_VAR = zeros((PARES*2, 72))
    for i in range(72):
    
        VCV = dot(X1T_CSP[:,:,i], X1T_CSP[:,:,i].T)
        X1T_VAR[:,i] = log(diag(VCV))
        
        VCV = dot(X2T_CSP[:,:,i], X2T_CSP[:,:,i].T)
        X2T_VAR[:,i] = log(diag(VCV))
        
        VCV = dot(X1V_CSP[:,:,i], X1V_CSP[:,:,i].T)
        X1V_VAR[:,i] = log(diag(VCV))
        
        VCV = dot(X2V_CSP[:,:,i], X2V_CSP[:,:,i].T)
        X2V_VAR[:,i] = log(diag(VCV))


    ## LDA

    m1 = mean(X1T_VAR, 1)
    m2 = mean(X2T_VAR, 1)

    S1 = cov(X1T_VAR)
    S2 = cov(X1V_VAR)
    Sw = S1 + S2

    B = dot(pinv(Sw), (m1-m2));
    b = dot(B.T, m1+m2) / 2;

    ## Verification on test Data

    acc1 = mean(dot(X1V_VAR.T, B) >= b)
    acc2 = 1 - mean(dot(X2V_VAR.T, B) >= b)
    acc_test = (acc1 + acc2) / 2
    
    return acc_test


if __name__ == "__main__":

    SUBJECT = 1
    classes = [1,2]
    acc_test = csp_lda(SUBJECT, classes)
    print('Test accuracy: ' + str(mean(att)*100))
