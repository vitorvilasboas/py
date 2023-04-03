#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 2 10:23:14 2020
@author: vboas
"""
import numpy as np
from bci_utils import extractEpochs
import math
path = '/mnt/dados/eeg_data/'

# IV2a: 576 trials (lh,rh,ft,to=144) | subj={1..9} | Fs=250Hz | cue=2s MI=3-6s end=7.5s | ch=22
ds = 'IV2a'; suj = 1
for suj in range(1,10):
    print(ds, 'A0'+str(suj))
    data, events, info = np.load(path+ds+'/npy/A0'+str(suj)+'.npy', allow_pickle=True)
    # epochs, labels = extractEpochs(data, events, 0, math.floor(info['fs']*7.5), [11,12,13,14])
    # for j in range(1,5): labels = np.where(labels == j+10, j, labels)
    epochs, labels = extractEpochs(data, events, math.floor(info['fs']*0.5), math.floor(info['fs']*2.5), [1,2,3,4])
    np.save(path+ds+'/npy_mi/A0'+str(suj)+'_mi', [epochs, labels, info])


# IV2b: 720 trials (lh,rh=360) | subj={1..9} | Fs=250Hz | cue=3s MI=4-7s end=8.5s | ch=3
ds = 'IV2b'; suj = 1
for suj in range(1,10):
    print(ds, 'B0'+str(suj))
    data, events, info = np.load(path+ds+'/npy/B0'+str(suj)+'.npy', allow_pickle=True)
    # epochs, labels = extractEpochs(data, events, 0, math.floor(info['fs']*8.5), [11,12])
    # for j in range(1,5): labels = np.where(labels == j+10, j, labels)
    epochs, labels = extractEpochs(data, events, 0, math.floor(info['fs']*4), [1,2])
    np.save(path+ds+'/npy_mi/B0'+str(suj)+'_mi', [epochs, labels, info])


# III3a: 240 trials (lh,rh,ft,to=60) | subj={'K3','K6','L1'} | Fs=250Hz | cue=3 MI=4-7s end=10s | ch=60
ds = 'III3a'; suj = 'K3'
for suj in ['K3','K6','L1']:
    print(ds, str(suj))
    data, events, info = np.load(path+ds+'/npy/'+str(suj)+'.npy', allow_pickle=True)
    # epochs, labels = extractEpochs(data, events, 0, math.floor(info['fs']*10), [11,12,13,14])
    # for j in range(1,5): labels = np.where(labels == j+10, j, labels)
    epochs, labels = extractEpochs(data, events, 0, math.floor(info['fs']*4), [1,2,3,4])
    np.save(path+ds+'/npy_mi/'+str(suj)+'_mi', [epochs, labels, info])



# III4a: 280 trials (rh,ft=140) | subj={'aa','al','av','aw','ay'} | Fs=100Hz | cue=0 MI=0-4s end=5s | ch=118
ds = 'III4a'; suj = 'aa'
for suj in ['aa','al','av','aw','ay']:
    print(ds, str(suj))
    data, events, info = np.load(path+ds+'/npy/'+str(suj)+'.npy', allow_pickle=True)
    # epochs, labels = extractEpochs(data, events, 0, math.floor(info['fs']*5), [2,3])
    epochs, labels = extractEpochs(data, events, 0, math.floor(info['fs']*4), [2,3])
    np.save(path+ds+'/npy_mi/'+str(suj)+'_mi', [epochs, labels, info])



# Lee19: 400 trials (lh,rh=200) | subj={1..54} | Fs=250Hz | cue=3s MI=3-7s end=10s | ch=62
ds = 'Lee19'; suj = 1
for suj in range(1,55):
    print(ds, 'S'+str(suj))
    data, events, info = np.load(path+ds+'/npy/S'+str(suj)+'.npy', allow_pickle=True)
    # epochs, labels = extractEpochs(data, events, 0, math.floor(info['fs']*10), [1,2])
    epochs, labels = extractEpochs(data, events, 0, math.floor(info['fs']*4), [1,2])
    np.save(path+ds+'/npy_mi/S'+str(suj)+'_mi', [epochs, labels, info])



e2a, l2a, i2a = np.load(path+'IV2a/npy_mi/A01_mi.npy', allow_pickle=True)
e2b, l2b, i2b = np.load(path+'IV2b/npy_mi/B03_mi.npy', allow_pickle=True)
e3a, l3a, i3a = np.load(path+'III3a/npy_mi/K3_mi.npy', allow_pickle=True)
e4a, l4a, i4a = np.load(path+'III4a/npy_mi/aw_mi.npy', allow_pickle=True)
e19, l19, i19 = np.load(path+'Lee19/npy_mi/S17_mi.npy', allow_pickle=True)


# z = lambda x: pow(x,2)
# z(2)
# list(map(z, [1,2,3,4]))