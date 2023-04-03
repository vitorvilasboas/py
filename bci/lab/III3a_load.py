#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:08:11 2019
@author: vboas
"""
import pandas as pd
import numpy as np
import mne
import math

def labeling(labels, trueLabels):
    labels = np.where(labels==4, 0, labels) # Start trial t=0
    labels = np.where(labels==5, 769, labels) # LH (classe 1) 
    labels = np.where(labels==6, 770, labels) # RH (classe 2) 
    labels = np.where(labels==7, 771, labels) # Foot (classe 3)
    labels = np.where(labels==8, 772, labels) # Tongue (classe 4)
    labels = np.where(labels==9, 783, labels) # Unknown
    idx = np.where(labels==0)
    labels[idx] = trueLabels
    return labels

def save_class_epoch(folder,filename,epochs,labels,ds):
    for label in [1,2,3,4]:
        idx = np.where(labels==label)
        epocas_i = epochs[idx]
        np.save(folder + 'npy/' + filename + '_' + ds + '_' + str(label), epocas_i) 

if __name__ == '__main__':
    Fs = 250.0
    Tmin, Tmax = 0, 10 # Start trial= 0; beep/cross= 2; Start cue=3; Start MI= 4; End MI= 7; End trial(break)= 10
    sample_min = int(math.floor(Tmin * Fs)) # amostra inicial (ex. 0)
    sample_max = int(math.floor(Tmax * Fs)) # amostra final (ex. 2500)
    
    folder = '/mnt/dados/datasets/BCICIII_3a/'
    filename = ['K3','K6','L1']
    mne.set_log_level('WARNING','DEBUG')
    
    # ds=0
    for ds in range(len(filename)):
        labels = np.ravel(pd.read_csv(folder + filename[ds] + '_truelabels.csv'))
        raw = mne.io.read_raw_gdf(folder + filename[ds] + '.gdf')
        raw.load_data()     
        data = raw.get_data()
        events = raw.find_edf_events()
        timestamps = np.delete(events[0],1,axis=1)
        cond = False
        for i in [1,2,3]: cond += (timestamps[:,1] == i)
        idx = np.where(cond)[0]
        timestamps = np.delete(timestamps,idx,axis=0)
         
        timestamps[:,1] = labeling(timestamps[:,1], labels)
        
        cond = False
        for i in [1,2,3,4]: cond += (timestamps[:,1] == i)
        idx = np.where(cond)[0]
        
        pos_begin = timestamps[idx,0] + sample_min
        pos_end = timestamps[idx,0] + sample_max
        n_epochs = len(timestamps[idx])
        n_channels = len(data)
        n_samples = sample_max - sample_min
        epochs = np.zeros([n_epochs, n_channels, n_samples])
        eeg_data = data[range(n_channels)]
        bad_epoch_list = []
        for i in range(n_epochs):
            epoch = eeg_data[:, pos_begin[i]:pos_end[i]]
            if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch # Check if epoch is complete   
            else:
                print('Incomplete epoch detected...',i)
                bad_epoch_list.append(i)
        labels = np.delete(labels, bad_epoch_list)
        epochs = np.delete(epochs, bad_epoch_list, axis=0)
        
        cond = False
        for i in [769,770,771,772,783]: cond += (timestamps[:,1] == i)
        classes = timestamps[np.where(cond)]
        
        idx = np.where(classes[:,1] == 783)[0]
        epochs_test = epochs[idx]
        labels_test = labels[idx]
        save_class_epoch(folder,filename[ds],epochs_test,labels_test,'E')
        
        idx = np.where(classes[:,1] != 783)[0]
        epochs_train = epochs[idx]
        labels_train = labels[idx]
        save_class_epoch(folder,filename[ds],epochs_train,labels_train,'T')
    
    
    
    
    DADOS = np.load(folder + 'npy/' + filename[1] + '_T_' + str(1) + '.npy') # teste de carregamento
    DADOS2 = np.load(folder + 'npy/' + filename[1] + '_E_' + str(1) + '.npy') # teste de carregamento