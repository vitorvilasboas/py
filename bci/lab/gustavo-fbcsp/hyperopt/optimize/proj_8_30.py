from hyperopt import hp

from sys import path

path.append('/home/gustavo/bci/code/bc3')
path.append('/home/gustavo/bci/code/hyperopt')

from fmin_path import fmin_path
from evaluate_bc3 import evaluate

def objective(args):
    
    
    #~ Print trial parameters
    
    n_bands, n_components, Clog, m = args
    
    print '\n\nn_bands: ', n_bands
    print 'n_components: ', n_components
    print 'Clog: ', Clog
    print 'm: ', m
    
    
    #~ Check for invalid combinations
    
    if n_bands >= m:
        return 100


    #~ Make trial dictionary
    
    args = {}
    
    args['n_bands'] = n_bands
    args['n_components'] = n_components
    args['Clog'] = Clog
    args['m'] = m
    
    args['t_start'] = 0.5
    args['t_end'] = 2.5
    args['fs'] = 100.
    args['fl'] = 8.
    args['fh'] = 30.
    
    
    #~ Set options
    
    options = {}
    options['sbcsp'] = True
    options['freq'] = True
    
    
    return evaluate(args, options)


if __name__ == "__main__":

    space = (
        hp.quniform		('n_bands',     	3, 		45, 	1),
        hp.quniform		('n_components', 	2, 		30, 	2),
        hp.quniform		('Clog', 			-8,		4,		1),
        hp.quniform		('m', 				20, 	45, 	1))
        
    # minimize the objective over the space
    best = fmin_path(objective, space, 1000, '/home/gustavo/bci/results/bc3/proj_8_30')

    print best
