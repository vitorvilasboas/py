from hyperopt import hp

from sys import path

path.append('/home/gustavo/bci/code/bc3')

from fmin_path import fmin_path
from evaluate_bc3 import evaluate

def objective(args):
    
    
    #~ Print trial parameters
    
    n_components, m = args
    
    print '\n\nn_components: ', n_components
    print 'm: ', m
    

    #~ Make trial dictionary
    
    args = {}
    
    args['n_components'] = n_components
    args['m'] = m
    
    args['t_start'] = 0.5
    args['t_end'] = 2.5
    args['fs'] = 100.
    args['fl'] = 8.
    args['fh'] = 30.
    
    
    #~ Set options
    
    options = {}
    options['sbcsp'] = False
    options['freq'] = True
    
    
    return evaluate(args, options)


if __name__ == "__main__":

    space = (
        hp.quniform		('n_components', 	2, 		30, 	2),
        hp.quniform		('m', 				20, 	45, 	1))
        
    # minimize the objective over the space
    best = fmin_path(objective, space, 1000, '/home/gustavo/bci/results/bc3/csp_lda_freq_8_30')

    print best
