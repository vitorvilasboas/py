from numpy import asarray, floor, mean
from filtering_proj import filtering

def averaging(X):
    
    W_Size = X.shape[2]
    H = 1
    L = 500
    N = int(floor((W_Size - L) / H + 1))
    
    XSL = [X[:,:,i*H:i*H+L] for i in range(N)]
    XSL = [filtering(XS) for XS in XSL]
    XA = mean(asarray(XSL), axis=0)
    
    return XA
