# from mne.decoding.csp import CSP
from sys import path
from numpy import concatenate, mean, ones, ravel, sign, sum, zeros
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

from load_data import load_data

path.append('../general')

from windowing import windowing
from filtering_iir import filtering
from mnecsp import CSP

def get_acc(SUBJECT, classes, args):

    fl, fh, m, n_components, Clog, n_bands, numtaps, atraso = args

    X = load_data(SUBJECT, classes, atraso)
    
    X = windowing(X, 2, 2.5, 4.5, 250, atraso)
    
    ## Set data format for CSP
    
    XT = concatenate(X[0])
    XV = concatenate(X[1])
    y = concatenate([zeros(72), ones(72)])

    ## Filtering
    f0 = 0.001
    XT = filtering(XT, fl=f0, fh=fh, numtaps=numtaps)
    XV = filtering(XV, fl=f0, fh=fh, numtaps=numtaps)
    
    ## Divide sub-bands
    
    n_bins = XT.shape[2]
    n_bands = 33
    step = n_bins / n_bands
    
    XT = [XT[:,:,i*step:(i+1)*step] for i in range(n_bands)]
    XV = [XV[:,:,i*step:(i+1)*step] for i in range(n_bands)]

    ## CSP
    
    csp = [CSP(n_components=6) for i in range(n_bands)]
    for i in range(n_bands):
        csp[i].fit(XT[i], y)
    XT_CSP = [csp[i].transform(XT[i]) for i in range(n_bands)]
    XV_CSP = [csp[i].transform(XV[i]) for i in range(n_bands)]

    ## LDA

    SCORE_V = zeros((144,n_bands))
    clf = [LinearDiscriminantAnalysis() for i in range(n_bands)]
    for i in range(n_bands):
        clf[i].fit(XT_CSP[i], y)
        SCORE_V[:,i] = ravel(clf[i].transform(XV_CSP[i]))

    acc_test = mean((sum(SCORE_V, axis=1) > 0) == y)

    return acc_test

if __name__ == "__main__":

    SUBJECT = 1
    classes = [1,2]
    acc_test = get_acc(SUBJECT, classes)
    print('Test accuracy: ' + str(acc_test*100))
