from numpy import concatenate, log, mean, ones, ravel, sign, std, sum, zeros
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sys import path
from time import time

path.append('../general')

from filtering_iir import filtering
from load_data import load_data
from mnecsp import CSP
from windowing import windowing


def get_acc(subject, classes, args):

    fl, fh, m, n_components, Clog, n_bands, numtaps, atraso = args
    numtaps = int(numtaps)
    n_bands = int(n_bands)

    X = load_data(subject, classes, atraso)
    
    X = windowing(X, 2, 2.5, 4.5, 250, atraso)

    # Set data format for CSP
    XT = concatenate(X[0])
    XV = concatenate(X[1])
    y = concatenate([zeros(72), ones(72)])

    # Divide sub-bands / Filtering
    fend = fh
    f0 = 0.0001  # f0=fl
    step = (fend-f0) / n_bands

    # FILTRA EM CADA SUB-BANDA
    XT = [filtering(XT, fl=f0+i*step, fh=f0+(i+1)*step, numtaps=numtaps) for i in range(n_bands)]
    XV = [filtering(XV, fl=f0+i*step, fh=f0+(i+1)*step, numtaps=numtaps) for i in range(n_bands)]

    # CSP
    csp = [CSP(n_components=n_components) for i in range(n_bands)]
    for i in range(n_bands):
        csp[i].fit(XT[i], y)
    XT_CSP = [csp[i].transform(XT[i]) for i in range(n_bands)]
    XV_CSP = [csp[i].transform(XV[i]) for i in range(n_bands)]

    # LDA
    SCORE_T = zeros((144, n_bands))
    SCORE_V = zeros((144, n_bands))
    clf = [LinearDiscriminantAnalysis() for i in range(n_bands)]
    for i in range(n_bands):
        clf[i].fit(XT_CSP[i], y)
        SCORE_T[:, i] = ravel(clf[i].transform(XT_CSP[i]))
        SCORE_V[:, i] = ravel(clf[i].transform(XV_CSP[i]))

    # Bayesian meta-classifier
    SCORE_T0 = SCORE_T[y==0,:]
    m0 = mean(SCORE_T0, axis=0)
    std0 = std(SCORE_T0, axis=0)
    
    SCORE_T1 = SCORE_T[y==1,:]
    m1 = mean(SCORE_T1, axis=0)
    std1 = std(SCORE_T1, axis=0)
    
    p0 = norm(m0, std0)
    p1 = norm(m1, std1)
    
    META_SCORE_T = log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
    META_SCORE_V = log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))

    # SVM on top of the meta-classifier
    svc = SVC(kernel="linear", C=10**Clog)
    svc.fit(META_SCORE_T, y)
    ans = svc.predict(META_SCORE_V)
    
    return mean(ans == y)


if __name__ == "__main__":

    subject = 1
    classes = [1,4]
    
    fl = 0
    fh = 51
    numtaps = 3
    n_components = 6
    Clog = -4
    n_bands = 33
    
    args = (fl, fh, numtaps, n_components, Clog, n_bands)
    
    t0 = time()
    acc_test = get_acc(subject, classes, args)
    print(time()-t0)
    
    print('Test accuracy: ' + str(acc_test*100))
