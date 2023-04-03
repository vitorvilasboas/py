from cPickle import load
from sklearn.model_selection import StratifiedKFold
from sys import stdout
from time import time


def evaluate(args, options):

	#~ Select method

	if options['sbcsp'] and options['proj']:
		from sbcsp_mc_freq import train_and_test
	elif options['sbcsp']:
		from sbcsp_mc_time import train_and_test
	elif options['proj']:
		from csp_lda_freq import train_and_test
	else:
		from csp_lda_time import train_and_test
		
	#~ Subjects to be evaluated

	#~ subjects = ['aa', 'al', 'av', 'aw', 'ay']
	subjects = ['aa']

	#~ Loop through subjects and folds

	acc_test = 0
	percentage = 0

	for subj_index in range(len(subjects)):

		file_path = "/home/gustavo/bci/data/bci3_iva/" + subjects[subj_index] + ".pickle"
		X, y = load(open(file_path))

		kf = StratifiedKFold(10, False, 0)
		
		t0 = time()
		
		for train_index, test_index in kf.split(X, y):
			
			acc_test += train_and_test(X, y, train_index, test_index, args)
			
			percentage += 100/(10*len(subjects))
			print(str(percentage) + '% '),
			stdout.flush()
			
		print('\nTime: ' + str(time() - t0))

	accuracy = acc_test*10/len(subjects)
	error = 100 - accuracy
	print('\nAccuracy: ' + str(accuracy))
	return error


if __name__ == "__main__":
    
    args = {}
    
    args['t_start'] = 0.5
    args['t_end'] = 2.5
    args['fs'] = 100.
    
    args['n_bands'] = 10
    args['fl'] = 5.
    args['fh'] = 35.
    
    args['n_components'] = 2
    args['Clog'] = -2
    
    args['m'] = 60
    #~ args['numtaps'] = 3

    options = {}
    options['sbcsp'] = True
    options['proj'] = True

    evaluate(args, options)
