# -*- coding: utf-8 -*-
import math
import numpy as np  # numpy - used for array and matrices operations
import scipy.signal as sp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
import scipy.linalg as lg


class CSP:
    def __init__(self, n_components=4, reg=None, log=True, cov_est="concat"):
        """Init of CSP."""
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.cov_est = cov_est
        self.filters_ = None
        self.patterns_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, epochs_data, y):
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)." % type(epochs_data))
        epochs_data = np.atleast_3d(epochs_data)

        e, c, t = epochs_data.shape
        # check number of epochs
        if e != len(y):
            raise ValueError("n_epochs must be the same for epochs_data and y")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("More than two different classes in the data.")
        if not (self.cov_est == "concat" or self.cov_est == "epoch"):
            raise ValueError("unknown covariance estimation method")
        if self.cov_est == "concat":  # concatenate epochs
            class_1 = np.transpose(epochs_data[y == classes[0]], [1, 0, 2]).reshape(c, -1)
            class_2 = np.transpose(epochs_data[y == classes[1]], [1, 0, 2]).reshape(c, -1)
            cov_1 = _regularized_covariance(class_1, reg=self.reg)
            cov_2 = _regularized_covariance(class_2, reg=self.reg)
        elif self.cov_est == "epoch":
            class_1 = epochs_data[y == classes[0]]
            class_2 = epochs_data[y == classes[1]]
            
            cov_1 = np.zeros((c, c))
            for t in class_1:
                cov_1 += _regularized_covariance(t, reg=self.reg)
            cov_1 /= class_1.shape[0]
            
            cov_2 = np.zeros((c, c))
            for t in class_2:
                cov_2 += _regularized_covariance(t, reg=self.reg)
            cov_2 /= class_2.shape[0]

        # normalize by trace
        cov_1 /= np.trace(cov_1)


        e, w = lg.eigh(cov_1, cov_1 + cov_2)
        n_vals = len(e)
        # Rearrange vectors
        ind = np.empty(n_vals, dtype=int)
        ind[::2] = np.arange(n_vals - 1, n_vals // 2 - 1, -1)
        ind[1::2] = np.arange(0, n_vals // 2)
        w = w[:, ind]  # first, last, second, second last, third, ...
        self.filters_ = w.T
        self.patterns_ = lg.pinv(w)

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in epochs_data])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, epochs_data, y=None):
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)." % type(epochs_data))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP decomposition.')
        if epochs_data.ndim == 2:
            pick_filters = self.filters_[:self.n_components]
            X = np.asarray([np.dot(pick_filters, epochs_data)])
        else:
            pick_filters = self.filters_[:self.n_components]
            X = np.asarray([np.dot(pick_filters, epoch) for epoch in epochs_data])
        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)
        if self.log:
            X = np.log(X)
        else:
            X -= self.mean_
            X /= self.std_
        return X


def _regularized_covariance(data, reg=None):
    if reg is None: cov = np.cov(data)
    return cov


class Filter:
    def __init__(self, fl, fh, srate, forder, filt_type='iir', band_type='band'):
        nyq = 0.5 * srate
        low = fl / nyq
        high = fh / nyq
        if filt_type == 'iir':
            self.b, self.a = sp.iirfilter(forder, [low, high], btype=band_type)
        elif filt_type == 'fir':
            self.b = sp.firwin(forder, [low, high], window='hamming', pass_zero=False)
            self.a = [1]

    def apply_filter(self, data_in):
        data_out = sp.filtfilt(self.b, self.a, data_in)
        return data_out
    

def nanCleaner(epoca):
    for i in range(epoca.shape[0]):
        bad_idx = np.isnan(epoca[i, :])
        epoca[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoca[i, ~bad_idx])
    return epoca


def extrairEpocas(data, eventos, classes, smin, smax):
    rotulos = eventos[:,1]
    cond = False
    for i in range(len(classes)): cond += (rotulos == classes[i])
    # cond é um vetor, cujo indice contém True se a posição correspondente em rotulos contém 1, 2, 3 ou 4
    idx = np.where(cond)[0] # contém as 288 indices que correspondem ao carimbo de uma das 4 classes 
    s = eventos[idx, 0] #contém os sample_stamp(posições) relacionadas ao inicio das 288 epocas das classes
    sBegin = s + smin # vetores que marcam a amostra que iniciam e finalizam cada época
    sEnd = s + smax
    data = data.T[range(22)]
    n_epochs = len(sBegin)
    n_channels = data.shape[0]
    n_samples = smax - smin
    epochs = np.zeros([n_epochs, n_channels, n_samples])
    labels = rotulos[idx] # vetor que contém os indices das 288 épocas das 4 classes
    bad_epoch_list = []
    for i in range(n_epochs):
        epoch = data[:, sBegin[i]:sEnd[i]]
        # Check if epoch is complete
        if epoch.shape[1] == n_samples:
            epochs[i, :, :] = epoch
        else:
            print('Incomplete epoch detected...')
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    return epochs, labels # retorna as 288 épocas e os indices de cada uma (labels)


if __name__ == '__main__':
    suj = 1
    Fs = 250.0
    classes = [1, 2]
    Tmin, Tmax = 2.5, 4.5 # a partir da dica
    smin = int(math.floor(Tmin * Fs))
    smax = int(math.floor(Tmax * Fs))
    f_low, f_high = 8., 30.
    f_order = 5
    csp_nei = 6

    path = '/mnt/dados/bci_tools/dset42a/npy/A0'
    dataT = np.load(open(path + str(suj) + 'T_data.npy', "rb"))
    dataE = np.load(open(path + str(suj) + 'E_data.npy', "rb"))
    eventT = np.load(open(path + str(suj) + 'T_event.npy', "rb"))
    eventE = np.load(open(path + str(suj) + 'E_event.npy', "rb"))
    
    epocasT, rotulosT = extrairEpocas(dataT, eventT, classes, smin, smax) # [288, 22, 500], [288]
    epocasE, rotulosE = extrairEpocas(dataE, eventE, classes, smin, smax)
    epocasT = nanCleaner(epocasT)
    epocasE = nanCleaner(epocasE)
    
    filtro = Filter(f_low, f_high, Fs, f_order, filt_type='iir', band_type='band')
    epocasT = filtro.apply_filter(epocasT)
    epocasE = filtro.apply_filter(epocasE)
    
    svc = LDA()
    csp = CSP(n_components=csp_nei, reg=None, log=True, cov_est='epoch')
    clf = Pipeline([('CSP', csp), ('SVC', svc)]) # executa uma sequencia de processamento com um classificador no final

    n_iter = 10
    cv = StratifiedShuffleSplit(n_iter, test_size=0.2, random_state=42)
    cross_scores = cross_val_score(clf, epocasT, rotulosT, cv=cv)
    print('Validação cruzada: {} {}'.format(round(cross_scores.mean() * 100, 2), round(cross_scores.std() *100, 2)))

    clf.fit(epocasT, rotulosT) # treinando o classificador
    
    auto_score = clf.score(epocasT, rotulosT) # como é auto validação, base de teste = base de treinamento
    print('Auto Validação: {}'.format(round(auto_score * 100, 2)))
    
    val_score = clf.score(epocasE, rotulosE) # agora sim, diferencio as bases na validação tradicional, treinamento != teste
    print('Validação {}'.format(round(val_score * 100, 2)))
