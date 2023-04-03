from numpy import ones
from scipy.signal import filtfilt, firwin

def filtering(X):

    fs = 250
    fl = 8.
    fh = 30.
    numtaps = 10
    
    b = firwin(numtaps, [fl/fs, fh/fs], window=('kaiser', 8), pass_zero=False)
    a = ones(1)

    XF = [[0,0],[0,0]]
    for i in range(2):
        for j in range(2):
            XF[i][j] = filtfilt(b, a, X[i][j], axis=-1)
            
    return XF
