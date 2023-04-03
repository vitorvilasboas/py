from cPickle import dump, load
from pprint import pprint

systems = [ 'csp_lda_proj_8_30',
            'csp_lda_filt_8_30',
            'proj_0_50',
            'filt_0_50',
            'proj_all',
            'filt_all',
            'proj_8_30']
            
#~ systems = [ 'filt_all',
            #~ 'proj_all',
            #~ 'proj_8_30']

#~ systems = [ 'proj_all' ]

#~ systems = [ 'mc',
            #~ 'mc_nbandas',
            #~ 'mc_nbandas_fl']

for i in range(len(systems)):

    path = '/home/gustavo/bci/results/bc3/' + systems[i]
    #~ path = '/home/gustavo/bci/results/bc4/' + systems[i]

    t = load(open(path))

    best = 100
    best_index = None
    n = 0

    for j in range(len(t.trials)):
        try:
            loss = t.trials[j]['result']['loss']
            if loss < best:
                best = loss
                best_index = j
            n += 1
        except:
            pass

    vals = t.trials[best_index]['misc']['vals']

    print('\n' + systems[i] + ' - Best between ' + str(n) + ' trials')
    print('\nValues:')
    for i in vals:
        print(str(i) + ': ' + str(vals[i][0]))
    print('\nLoss: ' + str(t.trials[best_index]['result']['loss']))
