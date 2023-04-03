# -*- coding: utf-8 -*-
# A frequency-weighted method combined with Common Spatial Patterns for electroencephalogram classification in brain–computer interface
#
# mean_log_spectre quase igual
# z pouco discriminativo
# w parece ruido

from struct import unpack

from matplotlib.pyplot import figure, plot, show
from numpy import arange, concatenate, log, mean, ones, reshape, transpose, zeros
from numpy.fft import fft
from scipy.linalg import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from load_data import load_data


def csp_lda(SUBJECT, classes):

    X = load_data(SUBJECT, classes)

    X1T = X[0][0]
    X2T = X[0][1]
    
    Y1T = fft(X1T, axis=1)
    Y2T = fft(X2T, axis=1)
    
    Y1T = log(abs(Y1T[:,:,0:500]) + 10**-10)
    Y2T = log(abs(Y2T[:,:,0:500]) + 10**-10)
    
    Y1T = reshape(transpose(Y1T, (1,2,0)), (500, 72*22))
    Y2T = reshape(transpose(Y2T, (1,2,0)), (500, 72*22))
    
    X = concatenate([Y1T, Y2T], 1).T
    y = concatenate([zeros(72*22), ones(72*22)])
    
    mean_log_spectre_1 = mean(Y1T.T, axis=0)
    mean_log_spectre_2 = mean(Y2T.T, axis=0)
    
    figure(figsize=(15,5))
    plot(mean_log_spectre_1)
    plot(mean_log_spectre_2)
    show()
    
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)

    w = abs(ravel(clf.coef_))

    figure(figsize=(15,5))
    plot(w)
    show()
    
    return 0


if __name__ == "__main__":

    subjectsT = arange(1,2)
    classesT = [[1,2]]
    
#    subjectsT = arange(1,10)
#    classesT = [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]

    att = []

    for SUBJECT in subjectsT:
        for classes in classesT:
            acc_test = csp_lda(SUBJECT, classes)
            att.append(acc_test)
        
#    print('Mean test accuracy: ' + str(mean(att)*100))
