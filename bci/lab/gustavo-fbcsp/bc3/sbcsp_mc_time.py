#~ from mne.decoding import CSP
from numpy import asarray, ceil, concatenate, log, mean, ones, ravel, sign, std, sum, zeros
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sys import stdout, path

path.append('/home/gustavo/bci/code/general')

from mycsp import CSP
from filtering_iir import filtering



def train_and_test(X, y, train_index, test_index, args):

    #~ Process parameters

    Clog = args['Clog']
    t_start = args['t_start']
    t_end = args['t_end']
    fs = args['fs']
    fl = args['fl']
    fh = args['fh']
    n_bands = int(args['n_bands'])
    n_components = int(args['n_components'])
    numtaps = int(args['numtaps'])

    #~ Windowing
    
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    X = X[:,:,W_Start:W_End]

    #~ Dividing training and test data
    
    XT = X[train_index]
    XV = X[test_index]
    yT = y[train_index]
    yV = y[test_index]
        
    #~ Filtering / Dividing sub-bands
    
    overlap = 2
    
    step = (fh-fl) / n_bands
    size = step * overlap
    
    XT = asarray([filtering(XT, fl=fl+i*step, fh=fl+i*step+size, numtaps=numtaps) for i in range(n_bands)])
    XV = asarray([filtering(XV, fl=fl+i*step, fh=fl+i*step+size, numtaps=numtaps) for i in range(n_bands)])
    
    ## CSP

    csp = [CSP(n_components=n_components) for i in range(n_bands)]
    for i in range(n_bands):
        csp[i].fit(XT[i], yT)
    XT_CSP = [csp[i].transform(XT[i]) for i in range(n_bands)]
    XV_CSP = [csp[i].transform(XV[i]) for i in range(n_bands)]

	## LDA

    SCORE_T = zeros((252,n_bands))
    SCORE_V = zeros((28,n_bands))
    clf = [LinearDiscriminantAnalysis() for i in range(n_bands)]
    for i in range(n_bands):
        clf[i].fit(XT_CSP[i], yT)
        SCORE_T[:,i] = ravel(clf[i].transform(XT_CSP[i]))
        SCORE_V[:,i] = ravel(clf[i].transform(XV_CSP[i]))

    ## Bayesian meta-classifier

    SCORE_T0 = SCORE_T[yT==1,:]
    m0 = mean(SCORE_T0, axis=0)
    std0 = std(SCORE_T0, axis=0)

    SCORE_T1 = SCORE_T[yT==2,:]
    m1 = mean(SCORE_T1, axis=0)
    std1 = std(SCORE_T1, axis=0)

    p0 = norm(m0, std0)
    p1 = norm(m1, std1)

    META_SCORE_T = log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
    META_SCORE_V = log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))

    ## SVM on top of the meta-classifier

    svc = SVC(kernel="linear", C=10**Clog)
    svc.fit(META_SCORE_T, yT)
    ans = svc.predict(META_SCORE_V)

    return mean(ans == yV)
    
    
if __name__ == "__main__":
    
    print('Not implemented')
