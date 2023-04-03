import sys
from matplotlib import pyplot as plt
from numpy import dot, linspace, ravel
from cPickle import dump

def get_weights(w, fl, fh, SUBJECT, classes):

    w = ravel(w)
    w = w / dot(w.T,w)
    
    struct = {}
    struct['w'] = w
    struct['SUBJECT'] = SUBJECT
    struct['classes'] = classes
    
    filename = 'freqs/S' + str(SUBJECT) + 'C' + str(classes)
    dump(struct, open(filename, 'w'))
