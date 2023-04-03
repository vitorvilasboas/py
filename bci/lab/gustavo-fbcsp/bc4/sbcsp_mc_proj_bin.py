# parameters: m, n_components, C
# best results: 65.94
# m = 150
# n_components = 6
# C = 10**-4

from numpy import concatenate, log, mean, ones, ravel, sign, std, sum, zeros
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sys import path

path.append('../general')

from mnecsp import CSP
from load_data import load_data
from windowing import windowing
from filtering_proj import filtering


def csp_lda(SUBJECT, classes):

    # fl, fh, m, n_components, Clog, n_bands, numtaps, atraso = args

    X = load_data(SUBJECT, classes, 0)
    
    X = windowing(X, 2, 2.5, 4.5, 250, 0)

    # Set data format for CSP
    XT = concatenate(X[0])
    XV = concatenate(X[1])
    y = concatenate([zeros(72), ones(72)])

    # Filtering
    m = 150
    XT = filtering(XT, fl=0., fh=94., m=m)
    XV = filtering(XV, fl=0., fh=94., m=m)

    # Divide sub-bands
    n_bands = m*2
    step = 1
    
    XT = [XT[:,:,i:(i+1)] for i in range(n_bands)]
    XV = [XV[:,:,i:(i+1)] for i in range(n_bands)]
    
    del XT[0]
    del XV[0]
    n_bands -= 1

    # CSP
    csp = [CSP(n_components=6) for i in range(n_bands)]
    for i in range(n_bands):
        csp[i].fit(XT[i], y)
        
    XT_CSP = [csp[i].transform(XT[i]) for i in range(n_bands)]
    XV_CSP = [csp[i].transform(XV[i]) for i in range(n_bands)]

    # LDA
    SCORE_T = zeros((144,n_bands))
    SCORE_V = zeros((144,n_bands))
    clf = [LinearDiscriminantAnalysis() for i in range(n_bands)]
    for i in range(n_bands):
        clf[i].fit(XT_CSP[i], y)
        SCORE_T[:,i] = ravel(clf[i].transform(XT_CSP[i]))
        SCORE_V[:,i] = ravel(clf[i].transform(XV_CSP[i]))

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
    svc = SVC(kernel="linear", C=10**-4)
    svc.fit(META_SCORE_T, y)
    ans = svc.predict(META_SCORE_V)
    
    return mean(ans == y)


if __name__ == "__main__":

    SUBJECT = 9
    classes = [1,2]
    acc_test = csp_lda(SUBJECT, classes)
    print('Test accuracy: ' + str(acc_test*100))
