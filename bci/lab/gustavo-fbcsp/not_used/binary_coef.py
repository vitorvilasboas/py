from numpy import *
from scipy.linalg import eig, norm, pinv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sin_basis import make_basis
from load_data import load_data


def csp_lda(SUBJECT, classes):

    X = load_data(SUBJECT, classes)
    
    X1T = X[0][0]
    X2T = X[0][1]
    X1V = X[1][0]
    X2V = X[1][1]

    t_0 = 2
    t_start = 2.5
    t_end = 4.5
    fs = 250

    W_Start = (t_start - t_0) * fs
    W_End = (t_end - t_0) * fs
    W_Size = W_End - W_Start
    
    X1T = X1T[:,W_Start:W_End,:]
    X2T = X2T[:,W_Start:W_End,:]
    X1V = X1V[:,W_Start:W_End,:]
    X2V = X2V[:,W_Start:W_End,:]

    ## Project on sinusoidal basis

    fl = 8.
    fh = 30.
    m = 40
    
    X0 = make_basis(W_Size, m, fl/fs, fh/fs)
    G0 = dot(pinv(dot(X0.T, X0)), X0.T)

    chan = 22
    ind = 72

    X1T_COEF = zeros((chan, 2*m, ind))
    X2T_COEF = zeros((chan, 2*m, ind))
    X1V_COEF = zeros((chan, 2*m, ind))
    X2V_COEF = zeros((chan, 2*m, ind))
    for i in range(chan):
        for j in range(ind):
            X1T_COEF[i,:,j] = dot(G0, X1T[i,:,j])
            X2T_COEF[i,:,j] = dot(G0, X2T[i,:,j])
            X1V_COEF[i,:,j] = dot(G0, X1V[i,:,j])
            X2V_COEF[i,:,j] = dot(G0, X2V[i,:,j])
    
    X1T = X1T_COEF
    X2T = X2T_COEF
    X1V = X1V_COEF
    X2V = X2V_COEF
    
    ## Extract means

    for i in range(chan):
        for j in range(ind):
            X1T[i,:,j] = X1T[i,:,j] - mean(X1T[i,:,j])
            X2T[i,:,j] = X2T[i,:,j] - mean(X2T[i,:,j])


    ## Make covariance matrices

    S1T = zeros((chan, chan))
    S2T = zeros((chan, chan))
    for i in range(ind):
        S1T = S1T + dot(X1T[:,:,i], X1T[:,:,i].T)
        S2T = S2T + dot(X2T[:,:,i], X2T[:,:,i].T)


    ## CSP

    [D, W] = eig(S1T, S1T + S2T)
    
    XT = concatenate([X1T, X2T], 2)
    XV = concatenate([X1V, X2V], 2)
    
    XT_CSP = zeros((chan, 2*m, 2*ind))
    XV_CSP = zeros((chan, 2*m, 2*ind))
    for i in range(2*ind):
        XT_CSP[:,:,i] = dot(W.T, XT[:,:,i])
        XV_CSP[:,:,i] = dot(W.T, XV[:,:,i])


    ## Log-variance
    
    XT_VAR = zeros((chan, 2*ind))
    XV_VAR = zeros((chan, 2*ind))
    for i in range(2*ind):
        XT_VAR[:,i] = log(diag(dot(XT_CSP[:,:,i], XT_CSP[:,:,i].T)))
        XV_VAR[:,i] = log(diag(dot(XV_CSP[:,:,i], XV_CSP[:,:,i].T)))


    ## LDA
    
    y = concatenate([zeros(ind), ones(ind)])

    clf = LinearDiscriminantAnalysis()
    clf.fit(XT_VAR.T, y)

    acc_test = mean(clf.predict(XV_VAR.T) == y)
    
    return acc_test


if __name__ == "__main__":

    SUBJECT = 1
    classes = [1,2]
    acc_test = csp_lda(SUBJECT, classes)
    print('Test accuracy: ' + str(mean(att)*100))
