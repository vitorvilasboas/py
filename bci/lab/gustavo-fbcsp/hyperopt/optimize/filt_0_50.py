from hyperopt import hp

from sys import path

path.append('/home/gustavo/bci/code/bc3')

from fmin_path import fmin_path
from evaluate_bc3 import evaluate

def objective(args):

    
    #~ Print trial parameters
    
    n_components, Clog, numtaps = args
    
    print '\n\nn_components: ', n_components
    print 'Clog: ', Clog
    print 'numtaps: ', numtaps
    

    #~ Make trial dictionary
    
    args = {}
    
    args['n_components'] = n_components
    args['Clog'] = Clog
    args['numtaps'] = numtaps
    
    args['n_bands'] = 24
    args['fl'] = 0.1
    args['fh'] = 50.
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
        hp.quniform		('n_components', 	2, 		30, 	2),
        hp.quniform		('Clog', 			-8,		4,		1),
        hp.quniform		('numtaps',			2, 	    10, 	1))
        
    # minimize the objective over the space
    best = fmin_path(objective, space, 100, '/home/gustavo/bci/results/bc3/filt_0_50')

    print best
