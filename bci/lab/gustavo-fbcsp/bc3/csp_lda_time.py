from mne.decoding import CSP
from numpy import mean, ravel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sys import stdout, path

path.append('/home/gustavo/bci/code/general')

from filtering_iir import filtering

def train_and_test(X, y, train_index, test_index, args):

    #~ Process parameters

    t_start = args['t_start']
    t_end = args['t_end']
    fs = args['fs']
    fl = args['fl']
    fh = args['fh']
    n_components = int(args['n_components'])
    numtaps = int(args['numtaps'])

    #~ Windowing
    
    W_Start = int(t_start * fs)
    W_End = int(t_end * fs)
    X = X[:,:,W_Start:W_End]

    #~ Filtering
    
    X = filtering(X, fs=fs, fl=fl, fh=fh, numtaps=numtaps)
    
    #~ Dividing training and test data
    
    XT = X[train_index]
    XV = X[test_index]
    yT = y[train_index]
    yV = y[test_index]

    ## CSP

    csp = CSP(n_components=n_components)
    csp.fit(XT, yT)
    XT_CSP = csp.transform(XT)
    XV_CSP = csp.transform(XV)

	## LDA
    
    clf = LinearDiscriminantAnalysis()
    clf.fit(XT_CSP, yT)
    ans_T = ravel(clf.predict(XT_CSP))
    ans_V = ravel(clf.predict(XV_CSP))
    
    return mean(ans_V == yV)


if __name__ == "__main__":

    #~ n_components = 6
    #~ fl = 8.
    #~ fh = 30.
    #~ 
    #~ args = n_components, fl, fh
        
    print('Not implemented')
