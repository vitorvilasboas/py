from hyperopt import hp

from sys import path

path.append('/home/gustavo/bci/code/bc3')

from fmin_path import fmin_path
from evaluate_bc3 import evaluate

def objective(args):

    
    #~ Print trial parameters
    
    n_bands, n_components, Clog, numtaps, fl, fh = args
    
    print '\n\nn_bands: ', n_bands
    print 'n_components: ', n_components
    print 'Clog: ', Clog
    print 'numtaps: ', numtaps
    print 'fl: ', fl
    print 'fh: ', fh
    

    #~ Make trial dictionary
    
    args = {}
    
    args['n_bands'] = n_bands
    args['n_components'] = n_components
    args['Clog'] = Clog
    args['numtaps'] = numtaps
    args['fl'] = fl
    args['fh'] = fh
    
    args['t_start'] = 0.5
    args['t_end'] = 2.5
    args['fs'] = 100.
    
    
    #~ Set options
    
    options = {}
    options['sbcsp'] = True
    options['freq'] = False
    
    
    return evaluate(args, options)


if __name__ == "__main__":

    space = (
        hp.quniform		('n_bands',     	3, 		60, 	1),
        hp.quniform		('n_components', 	2, 		30, 	2),
        hp.quniform		('Clog', 			-8,		4,		1),
        hp.quniform		('numtaps',			2, 	    10, 	1),
        hp.quniform		('fl', 				0,  	8,  	1),
        hp.quniform		('fh', 				30, 	50, 	1))
        
    # minimize the objective over the space
    best = fmin_path(objective, space, 100, '/home/gustavo/bci/results/bc3/filt_all')

    print best
