# Implementacao de CSP (from scratch)
# Diferencas em relacao a MNE:
#   Normalizacao das matrizes de covariancia
#   Normalizacao das features

from numpy import *
from scipy.linalg import eigh, eig

class CSP():

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):

        e, c, t = X.shape
        classes = unique(y)
        
        X0 = X[classes[0] == y,:,:]
        X1 = X[classes[1] == y,:,:]

        ## Sum up covariance matrix
        S0 = zeros((c, c))
        S1 = zeros((c, c))
        for i in range(e / 2):
        
            Si = cov(X0[i,:,:])
            S0 += Si / trace(Si)
            
            Si = cov(X1[i,:,:])
            S1 += Si / trace(Si)
            
        [D, W] = eigh(S0, S0 + S1)
#        [D, W] = eig(S0, S0 + S1)

        ind = empty(c, dtype=int)
        ind[0::2] = arange(c - 1, c // 2 - 1, -1)
        ind[1::2] = arange(0, c // 2)

        W = W[:, ind]
        
        self.filters_ = W.T[:self.n_components]

    def transform(self, X):

        XT = asarray([dot(self.filters_, epoch) for epoch in X])
        XVAR = log(mean(XT ** 2, axis=2))
#        XVAR -= mean(XVAR, axis=1)[:,newaxis]
        return XVAR
        
                
# Usage example: using only one combination of subject-class

if __name__ == "__main__":

    from numpy import ones, zeros
    from load_data import load_data
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
###    Load data
    SUBJECT = 1
    classes = [1,4]
    X = load_data(SUBJECT, classes)
    XT = concatenate(X[0])
    XV = concatenate(X[1])
    y = concatenate([zeros(72), ones(72)])
    
###    CSP
    n_components = 3
    csp = CSP(n_components=n_components)
    csp.fit(XT, y)
    XT_CSP = csp.transform(XT)
    XV_CSP = csp.transform(XV)

###    LDA
    clf = LinearDiscriminantAnalysis()
    clf.fit(XT_CSP, y)
    acc_test = mean(clf.predict(XV_CSP) == y)

    print acc_test
