from scipy.signal import filtfilt, firwin


def filtering(X, fs=250, fl=8., fh=30., numtaps=5):

    nf = fs / 2.  # Nyquist frequency

    b = firwin(numtaps, [fl/nf, fh/nf])  # Get coefficients

    XF = filtfilt(b, [1], X, axis=-1)  # Filter
    
    return XF
