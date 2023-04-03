from scipy.signal import filtfilt, iirfilter, lfilter, butter


def filtering(X, fs=250, fl=8., fh=30., numtaps=5.):

    nf = fs / 2.  # Nyquist frequency

    # [b,a] = iirfilter(numtaps, [fl/nf, fh/nf])  # Get coefficients
    # XF = filtfilt(b, a, X, axis=-1)  # Filter
    
    [b,a] = butter(numtaps, [fl/nf, fh/nf], btype='bandpass')
    XF = lfilter(b, a, X)
    
    return XF
