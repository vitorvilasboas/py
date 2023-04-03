# -*- coding: utf-8 -*-
# from __future__ import division
from numpy import arange, concatenate, transpose, mean, ones, zeros, cos, pi, sin, unique, dot, empty 
from numpy import asarray, log, dtype, fromfile, imag, real, ceil, ravel, sign, std, convolve, sum
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import filtfilt, iirfilter, lfilter, butter
from scipy.linalg import eigh, pinv
from time import time
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.fftpack import fft, ifft
from sys import path

path.append('./general')

from filtering_proj import filtering

def get_step(n_bins, n_bands, overlap):
	# Função usada para obter o step (intervalo de coeficientes entre duas bandas)
	# Estimativa inicial do step, que seria o número de bins sobre o número de bandas truncado em múltiplos de 2
	step = int((n_bins / n_bands / 2) * 2)
	size = step * overlap
	# Porém as vezes o resultado retornado faz com que a última banda tenha bins não existentes. Para corrigir isso usei um loop que verifica o último bin da 
    # última banda e se ele não for menor ou igual ao número de bins o step é reduzido por 2.
	while True:
		last_end = (n_bands-1) * step + size
		if last_end <= n_bins:
			break
		step -= 2
		size = step * overlap  # add by Vitor, cleison.py based
	return step


class CSP():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        e, c, t = X.shape
        classes = unique(y)
        
        X0 = X[classes[0] == y,:,:]
        X1 = X[classes[1] == y,:,:]

        # Sum up covariance matrix
        S0 = zeros((c, c))
        S1 = zeros((c, c))
        for i in range(int(e/2)): # add conversão int() ?
            S0 += dot(X0[i,:,:], X0[i,:,:].T)
            S1 += dot(X1[i,:,:], X1[i,:,:].T)

        [D, W] = eigh(S0, S0 + S1)

        ind = empty(c, dtype=int)
        ind[0::2] = arange(c - 1, c // 2 - 1, -1)
        ind[1::2] = arange(0, c // 2)
        
        W = W[:, ind]
        
        self.filters_ = W.T[:self.n_components]

    def transform(self, X):
        XT = asarray([dot(self.filters_, epoch) for epoch in X])
        XVAR = log(mean(XT ** 2, axis=2))
        
        return XVAR


def windowing(X, t_0, t_start, t_end, fs, atraso):
    W_Start = int((t_start - t_0) * fs)
    W_End = int((t_end - t_0) * fs)
    # print W_Start, W_End
    for i in range(2):
        for j in range(2):
            # X[i][j] = X[i][j][:, :, W_Start:W_End]
            Xa = X[i][j][:, :, W_Start:W_End]
            for cont in range(1, atraso + 1):
                Xb = X[i][j][:, :, W_Start - cont:W_End - cont]
                # print W_Start-cont, W_End-cont
                Xa = transpose(Xa, (1, 0, 2))
                Xb = transpose(Xb, (1, 0, 2))
                Xa = concatenate([Xa, Xb])
                Xa = transpose(Xa, (1, 0, 2))
            X[i][j] = Xa
    
    # print len(X),len(X[0]),len(X[0][0]),len(X[0][0][0]),len(X[0][0][0][0])
    
    return X


def load_data(SUBJECT, classes, atraso):
    folder = "/mnt/dados/bci_tools/dset42a/fdt/epocas_t2"
    X = [[], []]
    set = ['T_', 'E_']
    for i in range(2):
        for j in range(2):
            path = folder + '/A0' + str(SUBJECT) + set[j] + str(classes[i]) + '.fdt'
            fid = open(path, 'rb')
            data = fromfile(fid, dtype('f'))
            data = data.reshape((72, 1000, 22))
            X[j].append(transpose(data, (0,2,1)))
    return X


def sbcsp(subject, classes, args):

    fl, fh, m, n_components, Clog, n_bands, atraso = args
    m = int(m)
    n_bands = int(n_bands)

    X = load_data(subject, classes, atraso)

    X = windowing(X, 2, 2.5, 4.5, 250, atraso) 

    XT = concatenate(X[0])
    XV = concatenate(X[1])
    y = concatenate([zeros(72), ones(72)])

    # Filtering
    XT = filtering(XT, fl=fl, fh=fh, m=m)
    XV = filtering(XV, fl=fl, fh=fh, m=m)

    # Divide sub-bands
    n_bins = 2 * m 
    overlap = 2
    step = get_step(n_bins, n_bands, overlap)
    size = step * overlap  # tamanho fixo p/ todas sub bandas. overlap em 50%

    XT = [XT[:, :, i*step:i*step+size] for i in range(n_bands)]
    XV = [XV[:, :, i*step:i*step+size] for i in range(n_bands)] 

    # CSP
    csp = [CSP(n_components=n_components) for i in range(n_bands)]
    for i in range(n_bands):
        csp[i].fit(XT[i], y)

    XT_CSP = [csp[i].transform(XT[i]) for i in range(n_bands)]
    XV_CSP = [csp[i].transform(XV[i]) for i in range(n_bands)]

    # LDA
    SCORE_T = zeros((144, n_bands))
    SCORE_V = zeros((144, n_bands))
    clf = [LinearDiscriminantAnalysis() for i in range(n_bands)]
    for i in range(n_bands):
        clf[i].fit(XT_CSP[i], y)
        SCORE_T[:, i] = ravel(clf[i].transform(XT_CSP[i]))
        SCORE_V[:, i] = ravel(clf[i].transform(XV_CSP[i]))

    # Bayesian meta-classifier
    SCORE_T0 = SCORE_T[y == 0, :]
    m0 = mean(SCORE_T0, axis=0)
    std0 = std(SCORE_T0, axis=0)

    SCORE_T1 = SCORE_T[y == 1, :]
    m1 = mean(SCORE_T1, axis=0)
    std1 = std(SCORE_T1, axis=0)

    p0 = norm(m0, std0)
    p1 = norm(m1, std1)

    META_SCORE_T = log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
    META_SCORE_V = log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))

    # SVM on top of the meta-classifier
    svc = SVC(kernel="linear", C=10**Clog)
    svc.fit(META_SCORE_T, y)
    ans = svc.predict(META_SCORE_V)

    return mean(ans == y)


if __name__ == "__main__":
    
    fl = 0  		      	# frequência mínima
    fh = 50 		      	# frequência máxima
    m = 100				# n de sen e cos na base (resolução de frequência)
    n_components = 6 #4 é melhor 	# n de componentes; subconjunto de filtros espaciais associados > e < autovalores (em pares) CSP
    Clog = -4 			# parâmetro de regularização SVM
    n_bands = 33	        # n de sub-bandas
    atraso = 0			# n de atrasos amostrais (concatenação de janelas)
    	
    args = (fl, fh, m, n_components, Clog, n_bands, atraso)
    
    subjectsT = arange(1, 10)
    classesT = [[1, 2]]#, [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    att = zeros((len(subjectsT), len(classesT)))
    attime = zeros((len(subjectsT), len(classesT)))
    fullt = 0
    for SUBJECT, i in zip(subjectsT, range(10)):
        for classes, j in zip(classesT, range(7)):
            try:
                t0 = time()
                acc_test = sbcsp(SUBJECT, classes, args)
                t1 = time()
            except (KeyboardInterrupt, SystemExit):
                raise
            print(i + 1, classesT[j], ' Time:', round(t1 - t0, 2), ' Acc:', round(acc_test * 100, 2))
            att[i, j] = acc_test
            attime[i, j] = t1 - t0
            fullt += t1 - t0
            #stdout.flush()
    print('\nMean accuracy: ' + str(round(mean(att) * 100, 2)) + '%')
    print('Full time: ' + str(round(fullt, 2)) + 's')
    print('Mean time: ' + str(round(mean(attime), 2)) + 's')
    print('Mean time/subject: ' + str(round(fullt / 9, 2)) + 's')
    
    # subject = 9
	# classes = [1, 2]
	# t0 = time()
	# att = get_acc(subject, classes, args)
	# print(str(subject) + str(classes) + ' ' + str(round(time() - t0, 2)) + ' ' + str(round((att * 100), 2)))
