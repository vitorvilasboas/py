# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import mne
import math
import numpy as np
from scipy.io import loadmat

def extrairEpocas(data, eventos, classes, smin, smax):
    rotulos = eventos[:,2]
    cond = False
    for i in range(len(classes)): cond += (rotulos == classes[i])
    # cond é um vetor, cujo indice contém True se a posição correspondente em rotulos contém 1, 2, 3 ou 4
    idx = np.where(cond)[0] # contém as 288 indices que correspondem ao carimbo de uma das 4 classes 
    
    s = eventos[idx, 0] #contém os sample_stamp(posições) relacionadas ao inicio das 288 epocas das classes
    #for pos in s: print(pos)
    sBegin = s + smin # vetores que marcam a amostra que iniciam e finalizam cada época
    sEnd = s + smax
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

def rotulagem(rotulos, ds, trueLabels):
    rotulos = np.where(rotulos==1, 32766, rotulos) # Start of a new run/segment (after a break)
    rotulos = np.where(rotulos==2, 276, rotulos) # Idling EEG (eyes open)
    rotulos = np.where(rotulos==3, 277, rotulos) # Idling EEG (eyes closed)
    rotulos = np.where(rotulos==4, 1072, rotulos) # Eye movements
    rotulos = np.where(rotulos==5, 768, rotulos) # Start trial t=0
    if ds=='T': # if Training dataset (A0sT.gdf)
        rotulos = np.where(rotulos==6, 4, rotulos) # Tongue (classe 4)
        rotulos = np.where(rotulos==7, 3, rotulos) # Foot (classe 3)
        rotulos = np.where(rotulos==8, 2, rotulos) # RH (classe 2)
        rotulos = np.where(rotulos==9, 1, rotulos) # LH (classe 1)
        rotulos = np.where(rotulos==10, 1023, rotulos) # Rejected trial
    else: # if Evaluate dataset (A0sE.gdf) 
        rotulos = np.where(rotulos==7, 1023, rotulos) # Rejected trial
        j = 0
        for i in range(len(rotulos)): # rotula event_type a partir do vetor de truelabels
            if rotulos[i] == 6:
                rotulos[i] = trueLabels[j]
                j += 1
    return rotulos

def corrigeNaN(dados):
    for canal in range(dados.shape[0] - 1):
        this_chan = dados[canal]
        dados[canal] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(dados[canal])
        mediaCanal = np.nanmean(dados[canal])
        dados[canal, mask] = mediaCanal
    return dados

def nanCleaner(epoca):
    """Removes NaN from data by interpolation
    data_in : input data - np matrix channels x samples
    data_out : clean dataset with no NaN samples"""
    for i in range(epoca.shape[0]):
        bad_idx = np.isnan(epoca[i, :])
        epoca[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoca[i, ~bad_idx])
    return epoca
    
if __name__ == '__main__':
    mne.set_log_level('WARNING','DEBUG') 
    #folder = 'D:/bci_tools/dset42a/'
    folder = '/media/vboas/DADOS/bci_tools/dset42a/'
    ds = ['T','E']
    Fs = 250.0
    classes = [1, 2, 3, 4]
    Tmin, Tmax = 0, 4 # a partir da dica (máximo 4s qdo termina MI)
    smin = int(math.floor(Tmin * Fs))
    smax = int(math.floor(Tmax * Fs))
    
    for suj in range(1, 10): # (1, 10)
        for i in range(2): # (2)
            pathIN = folder + 'A0' + str(suj) + ds[i] + '.gdf'
            raw = mne.io.read_raw_gdf(pathIN)
            raw.load_data()
            dados = raw.get_data()
            #dados = corrigeNaN(dados) # Correção de NaN nos dados brutos
            dados = dados[range(22)] # 22 x 672528
            print(dados.shape)
            
            eventsGDF = raw.find_edf_events()
            eventos = eventsGDF[0][:,0:3] # 603 x 3
            trueLabels = np.ravel(loadmat(folder + 'true_labels/A0' + str(suj) + 'E.mat')['classlabel'])
            eventos[:,2] = rotulagem(eventos[:,2], ds[i], trueLabels)
            
            # Extrair todas as 288 épocas (72 por classe)
            epocas, rotulos = extrairEpocas(dados, eventos, classes, smin, smax) # [288, 22, 1000], [288]
            epocas = nanCleaner(epocas) # Correção de NaN nas épocas
            dados = dados.T
            pathOUT_data = folder + 'npyT/' + 'A0' + str(suj) + ds[i] + '_data'
            pathOUT_event = folder + 'npyT/' + 'A0' + str(suj) + ds[i] + '_event'
            pathOUT_epoch = folder + 'npyT/' + 'A0' + str(suj) + ds[i] + '_epochs'
            np.save(pathOUT_data, dados)
            np.save(pathOUT_event, eventos)
            np.save(pathOUT_epoch, epocas)
            
            # Extrair épocas de cada clsse específica
            for j in range(1, len(classes)+1):
                epocasi, rotulosi = extrairEpocas(dados.T, eventos, [j], smin, smax) # [72, 22, 1000], [72]
                pathOUT_epochsi = folder + 'npyT/epocas/' + 'A0' + str(suj) + ds[i] + '_' + str(j)
                np.save(pathOUT_epochsi, epocasi)
            