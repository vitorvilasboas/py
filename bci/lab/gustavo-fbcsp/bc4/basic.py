# from mne.decoding.csp import CSP
from numpy import concatenate, mean, ones, zeros
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sys import path
from time import time

path.append('../general')

from load_data import load_data
from windowing import windowing
# from filtering_fft import filtering
# from filtering_proj import filtering
from filtering_iir import filtering
from mnecsp import CSP


def get_acc(subject, classes, args):

    fl, fh, m, n_components, Clog, n_bands, numtaps, atraso = args

    X = load_data(subject, classes, atraso)
    
    # Filtering IIR ou FIR
    X[0][0] = filtering(X[0][0], fl=8., fh=30., numtaps=5)
    X[0][1] = filtering(X[0][1], fl=8., fh=30., numtaps=5)
    X[1][0] = filtering(X[1][0], fl=8., fh=30., numtaps=5)
    X[1][1] = filtering(X[1][1], fl=8., fh=30., numtaps=5)
    
    X = windowing(X, 2, 2.5, 4.5, 250, atraso)
    
    # Set data format for CSP
    XT = concatenate(X[0])
    XV = concatenate(X[1])
    y = concatenate([zeros(72), ones(72)])

    # Filtering IIR ou FIR
    # f0 = 0.0001
    # XT = filtering(XT, fl=8., fh=30., numtaps=5)
    # XV = filtering(XV, fl=8., fh=30., numtaps=5)

    # Filtering Proj
    # XT = filtering(XT, fl=8, fh=30, m=40)
    # XV = filtering(XV, fl=8, fh=30, m=40)

    # Filtering FFT
    # XT = filtering(XT, fl=8, fh=30)
    # XV = filtering(XV, fl=8, fh=30)

    # CSP
    csp = CSP(n_components=n_components)
    csp.fit(XT, y)
    XT_CSP = csp.transform(XT)
    XV_CSP = csp.transform(XV)

    # LDA
    clf = LinearDiscriminantAnalysis()
    clf.fit(XT_CSP, y)
    acc_test = mean(clf.predict(XV_CSP) == y)

    return acc_test


if __name__ == "__main__":

    subject = 1
    classes = [1,4]
    
    n_components = 6
    
    t0 = time()
    acc_test = get_acc(subject, classes, n_components)
    print(time()-t0)
    
    print('Test accuracy: ' + str(acc_test*100))
