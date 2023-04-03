#parameters: m, n_components, C

from sys import path
# from mne.decoding.csp import CSP
from numpy import concatenate, mean, ones, ravel, sign, sum, zeros
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

path.append('../general')

from load_data import load_data
from windowing import windowing
# from filtering_proj import filtering
from filtering_iir import filtering
from mnecsp import CSP

def get_acc(SUBJECT, classes, args):

    fl, fh, m, n_components, Clog, n_bands, numtaps, atraso = args

    X = load_data(SUBJECT, classes, atraso)

    X = windowing(X, 2, 2.5, 4.5, 250, atraso)

    # Set data format for CSP
    XT = concatenate(X[0])
    XV = concatenate(X[1])
    y = concatenate([zeros(72), ones(72)])

    # Filtering Proj
    # XT = filtering(XT, fl=0., fh=94., m=150)
    # XV = filtering(XV, fl=0., fh=94., m=150)

    # Filtering IIR
    f0 = 0.0001
    XT = filtering(XT, fl=f0, fh=fh, numtaps=numtaps)
    XV = filtering(XV, fl=f0, fh=fh, numtaps=numtaps)

    # Divide sub-bands
    n_bins = XT.shape[2]
    n_bands = 24
    step = n_bins / n_bands
    
    XT = [XT[:,:,i*step:(i+1)*step] for i in range(n_bands)]
    XV = [XV[:,:,i*step:(i+1)*step] for i in range(n_bands)]

    ## CSP
    csp = [CSP(n_components=8) for i in range(n_bands)]
    for i in range(n_bands):
        csp[i].fit(XT[i], y)
    XT_CSP = [csp[i].transform(XT[i]) for i in range(n_bands)]
    XV_CSP = [csp[i].transform(XV[i]) for i in range(n_bands)]

    ## LDA
    SCORE_T = zeros((144,n_bands))
    SCORE_V = zeros((144,n_bands))
    clf = [LinearDiscriminantAnalysis() for i in range(n_bands)]
    for i in range(n_bands):
        clf[i].fit(XT_CSP[i], y)
        SCORE_T[:,i] = ravel(clf[i].transform(XT_CSP[i]))
        SCORE_V[:,i] = ravel(clf[i].transform(XV_CSP[i]))

    ## SVM meta-classifier
    svc = SVC(kernel="linear", C=10**-3)
    svc.fit(SCORE_T, y)
    ans = svc.predict(SCORE_V)

    return mean(ans == y)

if __name__ == "__main__":

    SUBJECT = 1
    classes = [1,4]
    acc_test = csp_lda(SUBJECT, classes)
    print('Test accuracy: ' + str(acc_test*100))
