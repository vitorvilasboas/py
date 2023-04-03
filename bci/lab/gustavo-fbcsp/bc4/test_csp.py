from numpy import *
from numpy.fft import fft

from mne.decoding.csp import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.linalg import pinv

from sin_basis import make_basis
from load_data import load_data
from windowing import windowing

def csp_lda(SUBJECT, classes):

#    Load data and set meta-data
    X = load_data(SUBJECT, classes)
    t_0 = 2
    fs = 250
    n_channels = X[0][0].shape[1]
    n_epochs = X[0][0].shape[0]

#    Parameters
    t_start = 2.5
    t_end = 4.5
    fl = 8.
    fh = 30.
    m = 40
    n_components = 6
    
#    Windowing
    X = windowing(X, t_0, t_start, t_end, fs)

#    Project on sinusoidal basis
    PROJECT = 1
    if PROJECT:
        B = make_basis(W_Size, m, fl/fs, fh/fs)
        G0 = dot(pinv(dot(B.T, B)), B.T)
        XF = [[0,0],[0,0]]
        
        for i in range(2):
            for j in range(2):
                XF[i][j] = [dot(X[i][j][k,:,:], G0.T) for k in range(n_epochs)]
    else:
        FL = int(W_Size*fl/fs)
        FH = int(W_Size*fh/fs)
        FN = FH - FL
        XF = [[0,0],[0,0]]
        for i in range(2):
            for j in range(2):
                XF[i][j] = zeros((n_epochs, n_channels, 2*FN))
                for k in range(n_epochs):
                    for l in range(n_channels):
                        XFFT = fft(X[i][j][k,l,:])
                        XREAL = real(XFFT)[FL:FH]
                        XIMAG = imag(XFFT)[FL:FH]
                        XF[i][j][k,l,:] = concatenate([XREAL, XIMAG])

#    Train
###    Set data
    XT = concatenate(XF[0])
    XV = concatenate(XF[1])
    y = concatenate([zeros(72), ones(72)])

###    CSP
    csp = CSP(n_components=n_components)
    csp.fit(XT, y)
    XT_CSP = csp.transform(XT)
    XV_CSP = csp.transform(XV)

###    LDA
    clf = LinearDiscriminantAnalysis()
    clf.fit(XT_CSP, y)
    acc_test = mean(clf.predict(XV_CSP) == y)

    return acc_test


if __name__ == "__main__":

    SUBJECT = 1
    classes = [1,2]
    acc_test = csp_lda(SUBJECT, classes)
    print('Test accuracy: ' + str(mean(att)*100))
