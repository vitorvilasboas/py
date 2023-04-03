# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:54:05 2019
@author: Vitor Vilas-Boas
"""

import numpy as np

path_to_file = '/mnt/dados/datasets/BCI4_2a/duarte/A01T_data.npy'
path_to_labels_file = '/mnt/dados/datasets/BCI4_2a/duarte/A01T_events.npy'

dados = np.load(open(path_to_file, "rb"))
eventos = np.load(open(path_to_labels_file, "rb"))
eventos = np.load(path_to_labels_file)

loadedData = dados.T[:22]
#playback_labels = 

teste = iter(eventos)

atual = next(teste)
prox = next(teste)

# insert dummy column to fit mne event list format
t_events = np.insert(eventos, 1, values=0, axis=1)
t_events = t_events.astype(int)  # convert to integer

np.array