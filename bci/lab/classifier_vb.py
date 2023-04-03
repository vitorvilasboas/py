# -*- coding: utf-8 -*-
import math
import numpy as np
from scipy.io import loadmat
import scipy.signal as sp
import sys

sys.path.append('./reprod_rafael/')
from processor_vb import Learner, Filter

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
    
    learner = Learner()
    learner.design_LDA()
    learner.design_CSP(csp_nei)
    learner.assemble_learner()
    
    n_iter = 10
    test_perc = 0.2
    score = learner.cross_evaluate_set(epocasT, rotulosT, n_iter, test_perc)
    print('Validação cruzada: {}'.format(round(score * 100, 2)))

    learner.learn(epocasT, rotulosT)
    
    learner.evaluate_set(epocasT, rotulosT)
    autoscore = learner.get_results() # TREINAMENTO
    print('Auto Validação: {}'.format(round(autoscore * 100, 2)))
    
    learner.evaluate_set(epocasE, rotulosE)
    valscore = learner.get_results() # VALIDAÇÃO
    print('Validação {}'.format(round(valscore * 100,2)))
