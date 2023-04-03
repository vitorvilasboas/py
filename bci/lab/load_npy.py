# -*- coding: utf-8 -*-
import numpy as np

d0 = np.load('D:/bci_tools/eeg_data/A01Tduarte/data_cal.npy')
e0 = np.load('D:/bci_tools/eeg_data/A01Tduarte/events_cal.npy')

d1 = np.load('D:/bci_tools/dset42a/npy/A01T_data.npy')
e1 = np.load('D:/bci_tools/dset42a/npy/A01T_event.npy')

print(d0.shape)
print(d1.shape)

print(e0.shape)
print(e1.shape)

print(e0[10:20,1])
print(e1[10:20,1])