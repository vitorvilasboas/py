import cPickle

from hyperopt import base, tpe, fmin


def dump_trials(trials, path):

    print('Size of object: ' + str(len(trials)))
    print('Dumping')
    file = open(path, 'wb')
    cPickle.dump(trials, file)
    file.close()


def fmin_path(objective, space, max_evals, path):

	try:
		print 'Trying to pickle file'
		file = open(path, 'rb')
		trials = cPickle.load(file)
		file.close()
	except:
		print 'No trial file at specified path, creating new one'
		trials = base.Trials()
	else:
		print 'File found'

	try:
		print('Size of object: ' + str(len(trials)))
		best = fmin(objective, space=space, algo=tpe.suggest, max_evals=len(trials) + max_evals, trials=trials)
		dump_trials(trials, path)
	except:
		print('Exception raised')
		dump_trials(trials, path)
		raise

	return best
