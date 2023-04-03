# -*- coding: utf-8 -*-
from sys import stdout, path
from numpy import arange, mean, set_printoptions, zeros
from time import time
from timeit import timeit

path.append('../bc4')

# from sbcsp_mc_proj import get_acc		# SBCSP+SVM com Decomposição Sen/Cos (freq) na banda total
# from sbcsp_mc_fft import get_acc      	# SBCSP+SVM com FFT (freq) na banda total
# from sbcsp_mc_filt import get_acc  	# SBCSP+SVM com filtro IIR ou FIR (time) em cada sub-banda
# from sbcsp_svm import get_acc			# SBCSP+SVM com filtro IIR ou FIR (time) na banda total
# from sbcsp_simple import get_acc      # SBCSP com filtro IIR ou FIR (time) na banda total
from basic import get_acc   # CSP-LDA Clássico com filtro IIR ou FIR (time)


def evaluate(args):
	subjectsT = arange(1, 10)
	classesT = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
	att = zeros((len(subjectsT), len(classesT)))
	attime = zeros((len(subjectsT), len(classesT)))
	fullt = 0
	for SUBJECT, i in zip(subjectsT, range(10)):
		for classes, j in zip(classesT, range(1)):
			try:
				t0 = time()
				acc_test = get_acc(SUBJECT, classes, args)
				t1 = time()
			except (KeyboardInterrupt, SystemExit):
				raise
			print '--------------------------------------------------'
			print i + 1, classesT[j], ' Time:', round(t1 - t0, 2), ' Acc:', round(acc_test * 100, 2)
			# print i+1,';',classesT[j][0],';',classesT[j][1],';',round(t1 - t0, 2),';',round(acc_test * 100, 2)
			# print round(t1 - t0, 2),';',round(acc_test * 100, 2)
			att[i, j] = acc_test
			attime[i, j] = t1 - t0
			fullt += t1 - t0
			stdout.flush()
	print '--------------------------------------------------'
	print('\nMean accuracy: ' + str(round(mean(att) * 100, 2)) + '%')
	print('Full time: ' + str(round(fullt, 2)) + 's')
	print('Mean time/combination: ' + str(round(mean(attime), 2)) + 's')
	print('Mean time/subject: ' + str(round(fullt / 9, 2)) + 's')


	return 1 - mean(att)


if __name__ == "__main__":
	fl = 0  		        # frequência mínima
	fh = 51 	          	# frequência máxima
	m = 102				# n de sen e cos na base (resolução de frequência)
	n_components = 6 	# n de componentes; subconjunto de filtros espaciais associados > e < autovalores (em pares) CSP
	Clog = -4 			# parâmetro de regularização SVM
	n_bands = 33	        # n de sub-bandas
	numtaps = 3			# ordem dos filtros IIR e FIR ??
	atraso = 0			# n de atrasos amostrais (concatenação de janelas)

	args = (fl, fh, m, n_components, Clog, n_bands, numtaps, atraso)
	evaluate(args)

	# subject = 9
	# classes = [1, 2]
	# t0 = time()
	# att = get_acc(subject, classes, args)
	# print(str(subject) + str(classes) + ' ' + str(round(time() - t0, 2)) + ' ' + str(round((att * 100), 2)))
