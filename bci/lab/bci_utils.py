# -*- coding: utf-8 -*-
import re
import os
import mne
import math
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time, sleep
from sklearn.svm import SVC
from scipy.io import loadmat
from scipy.stats import norm
from datetime import datetime
from scipy.fftpack import fft
from scipy.linalg import eigh
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold

np.seterr(divide='ignore', invalid='ignore')



def iii3a_to_omi(suj, path, channels=60):
    """ Dataset description/Meta-info MNE (Linux) (by vboas):
        1=Beep (accustic stimulus, BCI experiment)
        2=Cross on screen (BCI experiment)
        3=Rejection of whole trial
        4=Start of Trial, Trigger at t=0s
        5=769 class1, Left hand - cue onset (BCI experiment)
        6=770 class2, Right hand - cue onset (BCI experiment)
        7=771 class3, Foot, towards Right - cue onset (BCI experiment)
        8=772 class4, Tongue - cue onset (BCI experiment)
        9=783 cue unknown/undefined (used for BCI competition) 
    """
    mne.set_log_level('WARNING','DEBUG')
    raw = mne.io.read_raw_gdf('/mnt/dados/eeg_data/BCI3_3a/gdf/K3.gdf')
    truelabels = np.ravel(pd.read_csv('/mnt/dados/eeg_data/BCI3_3a/gdf/true_labels/trues_K3.csv'))
    raw = mne.io.read_raw_gdf(path + '/gdf/' + suj + '.gdf')
    raw.load_data()
    data = raw.get_data() # [channels x samples]
    data = data[:channels]
    # data = corrigeNaN(data) # Correção de NaN nos dados brutos
    events_raw = raw.find_edf_events()
    ev = np.delete(events_raw[0],1,axis=1) # elimina coluna de zeros
    truelabels = np.ravel(pd.read_csv(path + '/gdf/true_labels/trues_' + suj + '.csv'))
       
    cond = False
    for i in [1, 2, 3]: cond += (ev[:,1] == i)
    idx = np.where(cond)[0]
    ev = np.delete(ev, idx, axis=0)
    
    ev[:,1] = np.where(ev[:,1]==4, 0, ev[:,1]) # Labeling Start trial t=0
    
    idx = np.where(ev[:,1]!=0)
    ev[idx,1] = truelabels  

    # cond = False
    # for i in [5,6,7,8,9]: cond += (ev[:,1] == i)
    # idx = ev[np.where(cond)]
    # ev[np.where(cond),1] = truelabels
    
    info = {'fs': 250, 'class_ids': [1, 2, 3, 4], 'trial_tcue': 3.0,
            'trial_tpause': 8.0, 'trial_mi_time': 5.0, 'trials_per_class': 60,
            'eeg_channels': 60, 'ch_labels': raw.ch_names,
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
    
    omi_data = [ data, ev, info ]
    with open(path + '/omi/' + suj + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)


def iii4a_to_omi(suj, path):
    mat = loadmat(path + '/mat/' + suj + '.mat')
    data = mat['cnt'].T # 0.1 * mat['cnt'].T # convert to uV
    pos = mat['mrk'][0][0][0][0]
    true_mat = loadmat(path + '/mat/true_labels/trues_' + suj + '.mat')
    true_y = np.ravel(true_mat['true_y']) # RH=1 Foot=2
    true_y = np.where(true_y == 2, 3, true_y) # RH=1 Foot=3
    # true_test_idx = np.ravel(true_mat['test_idx'])
    events = np.c_[pos, true_y]
    # data = corrigeNaN(data)
    # data = np.asarray([ np.nan_to_num(dt) for dt in data ])
    # data = np.asarray([ np.ravel(pd.DataFrame(dt).fillna(pd.DataFrame(dt).mean())) for dt in data ])
    info = {'fs': 100, 'class_ids': [1, 3], 'trial_tcue': 0,
            'trial_tpause': 5.0, 'trial_mi_time': 4.5, 'trials_per_class': 140,
            'eeg_channels': 118, 'ch_labels': mat['nfo']['clab'],
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
    
    omi_data = [ data, events, info ]
    with open(path + '/omi/' + suj + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)


def iv2a_to_omi(suj, path, channels=25):
    """ Dataset description MNE (Linux) (by vboas): more info in http://bbci.de/competition/iv/desc_2a.pdf
            Meta-info (Training data _T):
             	1=1023 (rejected trial)
             	2=768 (start trial)
             	3=1072 (Unknown/ Eye Moviments)
             	4=769 (Class 1 - LH - cue onset)
             	5=770 (Class 2 - RH - cue onset)
             	6=771 (Class 3 - Foot - cue onset)
             	7=772 (Class 4 - Tongue - cue onset)
             	8=277 (Eye closed) [suj 4 = 32766 (Start a new run)]
             	9=276 (Eye open)   [suj 4 = None ]
             	10=32766 (Start a new run) [suj 4 = None ]
            Meta-info (Test data _E):
             	1=1023 (rejected trial)
             	2=768 (start trial)
             	3=1072 (Unknown/ Eye Moviments)
             	4=783 (Cue unknown/undefined)
             	5=277 (Eye closed)
             	6=276 (Eye open)
             	7=32766 (Start a new run)
        """
    mne.set_log_level('WARNING','DEBUG')
    dataT, dataV, eventsT, eventsV = [],[],[],[]
    channels_labels = None
    for ds in ['T', 'E']:
        # raw = mne.io.read_raw_gdf('/mnt/dados/eeg_data/BCI4_2a/gdf/A01T.gdf')
        raw = mne.io.read_raw_gdf(path + '/gdf/A0' + str(suj) + ds + '.gdf')
        raw.load_data()
        data = raw.get_data() # [channels x samples]
        data = data[:channels]
        # data = self.corrigeNaN(data) # Correção de NaN nos dados brutos
        raw_events = raw.find_edf_events()
        ev = np.delete(raw_events[0], 1, axis=1) # elimina coluna de zeros do MNE
        truelabels = np.ravel(loadmat(path + '/gdf/true_labels/A0' + str(suj) + 'E.mat' )['classlabel']) # Loading true labels to use in evaluate files (E)
        ev = np.delete(ev,np.where(ev[:,1]==1), axis=0) # Rejected trial
        ev = np.delete(ev,np.where(ev[:,1]==3), axis=0) # Eye movements / Unknown
        ev[:,1] = np.where(ev[:,1]==2, 0, ev[:,1]) # Start trial t=0
        if ds=='T':
            ev[:,1] = np.where(ev[:,1]==4, 1, ev[:,1]) # LH (classe 1) 
            ev[:,1] = np.where(ev[:,1]==5, 2, ev[:,1]) # RH (classe 2) 
            ev[:,1] = np.where(ev[:,1]==6, 3, ev[:,1]) # Foot (classe 3)
            ev[:,1] = np.where(ev[:,1]==7, 4, ev[:,1]) # Tongue (classe 4) 
            ev = np.delete(ev,np.where(ev[:,1]==8), axis=0) # Idling EEG (eyes closed) 
            ev = np.delete(ev,np.where(ev[:,1]==9), axis=0) # Idling EEG (eyes open) 
            ev = np.delete(ev,np.where(ev[:,1]==10), axis=0) # Start of a new run/segment (after a break)
            dataT, eventsT = data, ev
        else:
            ev = np.delete(ev,np.where(ev[:,1]==5), axis=0) # Idling EEG (eyes closed) 
            ev = np.delete(ev,np.where(ev[:,1]==6), axis=0) # Idling EEG (eyes open)
            ev = np.delete(ev,np.where(ev[:,1]==7), axis=0) # Start of a new run/segment (after a break)
            ev[np.where(ev[:,1]==4),1] = truelabels # muda padrão dos rotulos desconhecidos de 4 para [1,2,3,4]
            dataV, eventsV = data, ev
        channels_labels = raw.ch_names
    all_data = np.c_[dataT, dataV]    
    new_events = np.copy(eventsV)
    new_events[:,0] += len(dataT.T) # eventsV pos + last dataT pos (eventsT is continued by eventsV)
    all_events = np.r_[eventsT, new_events]
    info = {'fs': 250, 'class_ids': [1, 2, 3, 4], 'trial_tcue': 2.0,
            'trial_tpause': 7.0, 'trial_mi_time': 5.0, 'trials_per_class': 144,
            'eeg_channels': 22, 'ch_labels': channels_labels,
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
    omi_data = [ all_data, all_events, info ]
    with open(path + '/omi/A0' + str(suj) + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)
    

def iv2b_to_omi(suj, path, channels=3):
    """ Dataset Description (by vboas): more info in http://bbci.de/competition/iv/desc_2b.pdf
        01T e 02T (without feedback)
    		10 trial * 2 classes * 6 runs * 2 sessions = 240 trials (120 per class)
    		Cross t=0 (per 3s)
    		beep t=2s
    		cue t=3s (per 1.25s)
    		MI t=4s (per 3s)
    		Pause t=7s (per 1.5-2.5s)
    		EndTrial t=8.5-9.5
    	03T, 04E e 05E (with feedback)
    		10 trial * 2 classes * 4 runs * 3 sessions = 240 trials (120 per class)
    		Smiley(grey) t=0 (per 3.5s)
    		beep t=2s
    		cue t=3s (per 4.5s)
    		MI (Feedback períod) t=3.5s (per 4s)
    		Pause t=7.5s (per 1-2s)
    		EndTrial t=8.5-9.5
    	Meta-info 01T e 02T:
    		1=1023 (rejected trial)
    		2=768 (start trial)
    		3=769 (Class 1 - LH - cue onset)
    		4=770 (Class 2 - RH - cue onset)
    		5=277 (Eye closed)
    		6=276 (Eye open)
    		7=1081 (Eye blinks)
    		8=1078 (Eye rotation)
    		9=1077 (Horizontal eye movement)
    		10=32766 (Start a new run) *(to B0102T == 5)
    		11=1078 (Vertical eye movement)			
    	Meta-info 03T:
    		1=781 (BCI feedback - continuous)
    		2=1023 (rejected trial)
    		3=768 (start trial)
    		4=769 (Class 1 - LH - cue onset)
    		5=770 (Class 2 - RH - cue onset)
    		6=277 (Eye closed)
    		7=276 (Eye open)
    		8=1081 (Eye blinks)
    		9=1078 (Eye rotation)
    		10=1077 (Horizontal eye movement)
    		11=32766 (Start a new run)
    		12=1078 (Vertical eye movement)
    	Meta-info 04E e 05E:
    		1=781 (BCI feedback - continuous)
    		2=1023 (rejected trial)
    		3=768 (start trial)
    		4=783 (Cue unknown/undefined)
    		5=277 (Eye closed)
    		6=276 (Eye open)
    		7=1081 (Eye blinks)
    		8=1078 (Eye rotation)
    		9=1077 (Horizontal eye movement)
    		10=32766 (Start a new run)
    		11=1078 (Vertical eye movement)
    """
    mne.set_log_level('WARNING','DEBUG')
    DT, EV = [], []
    for ds in ['01T','02T','03T','04E','05E']:
        # raw = mne.io.read_raw_gdf('/mnt/dados/eeg_data/BCI4_2b/gdf/B0101T.gdf')
        raw = mne.io.read_raw_gdf(path + '/gdf/B0' + str(suj) + ds + '.gdf')
        raw.load_data()
        data = raw.get_data() # [channels x samples]
        data = data[:channels]
        # data = corrigeNaN(data) # Correção de NaN nos dados brutos
        ev = raw.find_edf_events()
        ev = np.delete(ev[0],1,axis=1) # elimina coluna de zeros
        truelabels = np.ravel(loadmat(path + '/gdf/true_labels/B0' + str(suj) + ds + '.mat')['classlabel'])

        if ds in ['01T','02T']:
            for rm in range(5,12): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
            ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete rejected trials
            ev[:,1] = np.where(ev[:,1]==2, 0, ev[:,1]) # altera label start trial de 2 para 0
            ev[:,1] = np.where(ev[:,1]==3, 1, ev[:,1]) # altera label cue LH de 3 para 1
            ev[:,1] = np.where(ev[:,1]==4, 2, ev[:,1]) # altera label cue RH de 4 para 2
        elif ds=='03T': 
            for rm in range(6,13): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
            ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # delete rejected trials
            ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete feedback continuous
            ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1]) # altera label start trial de 3 para 0
            ev[:,1] = np.where(ev[:,1]==4, 1, ev[:,1]) # altera label cue LH de 4 para 1
            ev[:,1] = np.where(ev[:,1]==5, 2, ev[:,1]) # altera label cue RH de 5 para 2
        else:
            for rm in range(5,12): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
            ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # delete rejected trials
            ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete feedback continuous
            ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1]) # altera label start trial de 3 para 0
            ev[np.where(ev[:,1]==4),1] = truelabels #rotula momento da dica conforme truelabels
        
        DT.append(data)
        EV.append(ev)
        
    # Save a unique npy file with all datasets
    soma = 0
    for i in range(1,len(EV)): 
        soma += len(DT[i-1].T)
        EV[i][:,0] += soma
        
    all_data = np.c_[DT[0],DT[1],DT[2],DT[3],DT[4]]
    all_events = np.r_[EV[0],EV[1],EV[2],EV[3],EV[4]]
    
    info = {'fs': 250, 'class_ids': [1, 2], 'trial_tcue': 3.0,
            'trial_tpause': 8.0, 'trial_mi_time': 5.0, 'trials_per_class': 360,
            'eeg_channels': 3, 'ch_labels': {'EEG1':'C3', 'EEG2':'Cz', 'EEG3':'C4'},
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
    
    omi_data = [ all_data, all_events, info ]
    with open(path + '/omi/B0' + str(suj) + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)


def cl_to_omi(suj, path):
    data = np.load(path + '/original/orig_CL_LF_data.npy').T
    events = np.load(path + '/original/orig_CL_LF_events.npy').astype(int)
    events[:,1] = np.where(events[:,1] == 2, 3, events[:,1]) # LH=1, FooT=3
    
    info = {'fs': 125, 'class_ids': [1, 3], 'trial_tcue': 2.0,
            'trial_tpause': 10.0, 'trial_mi_time': 8.0, 'trials_per_class': 24,
            'eeg_channels': 16, 'ch_labels': None,
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
    omi_data = [ data, events, info ]
    with open(path + '/omi/CL_LF.omi', 'wb') as handle: pickle.dump(omi_data, handle)
    
    data = np.load(path + '/original/orig_CL_LR_data.npy').T
    events = np.load(path + '/original/orig_CL_LR_events.npy').astype(int)
    info = {'fs': 125, 'class_ids': [1, 2], 'trial_tcue': 2.0,
            'trial_tpause': 10.0, 'trial_mi_time': 8.0, 'trials_per_class': 50,
            'eeg_channels': 16, 'ch_labels': None,
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
    omi_data = [ data, events, info ]
    with open(path + '/omi/CL_LR.omi', 'wb') as handle: pickle.dump(omi_data, handle)


def twl_to_omi(suj, path, channels=8):
    """ Dataset description/Meta-info MNE (Linux) (by vboas):
        1=Cross on screen (BCI experiment) 
        2=Feedback (continuous) - onset (BCI experiment)
        3=768 Start of Trial, Trigger at t=0s
        4=783 Unknown
        5=769 class1, Left hand - cue onset (BCI experiment)
        6=770 class2, Right hand - cue onset (BCI experiment)
    """
    mne.set_log_level('WARNING','DEBUG')
    dataT, dataV, eventsT, eventsV = [],[],[],[]
    for ds in ['S1','S2']:
        raw = mne.io.read_raw_gdf(path + '/gdf/' + suj + '_' + ds + '.gdf')
        raw.load_data()
        data = raw.get_data() # [channels x samples]
        data = data[:channels]
        # data = corrigeNaN(data) # Correção de NaN nos dados brutos
        events_raw = raw.find_edf_events()
        ev = np.delete(events_raw[0],1,axis=1) # elimina coluna de zeros
        
        ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # elimina marcações inuteis (cross on screen)
        ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # elimina marcações inuteis (feedback continuous)
        ev = np.delete(ev,np.where(ev[:,1]==4),axis=0) # elimina marcações inuteis (unknown)
        ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1])
        ev[:,1] = np.where(ev[:,1]==5, 1, ev[:,1]) # altera label lh de 5 para 1
        ev[:,1] = np.where(ev[:,1]==6, 2, ev[:,1]) # altera label rh de 6 para 2
        
        if ds == 'S1': dataT, eventsT = data, ev
        else: dataV, eventsV = data, ev
    
    all_data = np.c_[dataT, dataV]    
    new_events = np.copy(eventsV)
    new_events[:,0] += len(dataT.T) # eventsV pos + last dataT pos (eventsT is continued by eventsV)
    all_events = np.r_[eventsT, new_events]
    
    info = {'fs': 250, 'class_ids': [1, 2], 'trial_tcue': 3.0,
            'trial_tpause': 10.0, 'trial_mi_time': 7.0, 'trials_per_class': 20,
            'eeg_channels': 8, 'ch_labels': {'EEG1':'Cz', 'EEG2':'Cpz', 'EEG3':'C1', 'EEG4':'C3', 'EEG5':'CP3', 'EEG6':'C2', 'EEG7':'C4', 'EEG8':'CP4'},
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
    omi_data = [ all_data, all_events, info ]
    with open(path + '/omi/' + suj + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)


def iii3a_labeling(ev, suj, trueLabels):
    """ Dataset description/Meta-info MNE (Linux) (by vboas):
        1=Beep (accustic stimulus, BCI experiment)
        2=Cross on screen (BCI experiment)
        3=Rejection of whole trial
        4=Start of Trial, Trigger at t=0s
        5=769 class1, Left hand - cue onset (BCI experiment)
        6=770 class2, Right hand - cue onset (BCI experiment)
        7=771 class3, Foot, towards Right - cue onset (BCI experiment)
        8=772 class4, Tongue - cue onset (BCI experiment)
        9=783 cue unknown/undefined (used for BCI competition) 
    """
    cond = False
    for i in [1, 2, 3]: cond += (ev[:,1] == i)
    idx = np.where(cond)[0]
    ev = np.delete(ev, idx, axis=0)
    
    idx = np.where(ev[:,1]==4)
    ev[idx,1] = trueLabels  # Labeling Start trial t=0

    cond = False
    for i in [5,6,7,8,9]: cond += (ev[:,1] == i)
    idx = ev[np.where(cond)]
    ev[np.where(cond),1] = trueLabels + 768
    
    return ev


def iv2a_labeling(labels, ds, suj, trues, so='lnx'): 
    # normaliza rotulos de eventos conforme descrição oficial do dataset
    if so == 'lnx':
        """ Dataset description MNE (Linux) (by vboas): more info in http://bbci.de/competition/iv/desc_2a.pdf
            Meta-info (Training data _T):
             	1=1023 (rejected trial)
             	2=768 (start trial)
             	3=1072 (Unknown/ Eye Moviments)
             	4=769 (Class 1 - LH - cue onset)
             	5=770 (Class 2 - RH - cue onset)
             	6=771 (Class 3 - Foot - cue onset)
             	7=772 (Class 4 - Tongue - cue onset)
             	8=277 (Eye closed) [suj 4 = 32766 (Start a new run)]
             	9=276 (Eye open)   [suj 4 = None ]
             	10=32766 (Start a new run) [suj 4 = None ]
            Meta-info (Test data _E):
             	1=1023 (rejected trial)
             	2=768 (start trial)
             	3=1072 (Unknown/ Eye Moviments)
             	4=783 (Cue unknown/undefined)
             	5=277 (Eye closed)
             	6=276 (Eye open)
             	7=32766 (Start a new run)
        """
        labels = np.where(labels==1, 1023, labels) # Rejected trial
        labels = np.where(labels==2, 768, labels) # Start trial t=0
        labels = np.where(labels==3, 1072, labels) # Eye movements / Unknown
        
        if ds=='T': # if Training dataset (A0sT.gdf) 
            labels = np.where(labels==4, 769, labels) # LH (classe 1) 
            labels = np.where(labels==5, 770, labels) # RH (classe 2) 
            labels = np.where(labels==6, 771, labels) # Foot (classe 3)
            labels = np.where(labels==7, 772, labels) # Tongue (classe 4) 
            
            if suj == 4:
                labels = np.where(labels==8, 32766, labels) # Start of a new run/segment (after a break)
            else:
                labels = np.where(labels==8, 277, labels) # Idling EEG (eyes closed) 
                labels = np.where(labels==9, 276, labels) # Idling EEG (eyes open) 
                labels = np.where(labels==10, 32766, labels) # Start of a new run/segment (after a break)
            
            for i in range(0, len(labels)): # rotula dica conforme tarefa
                if labels[i]==768: # rotula [1 a 4] o inicio da trial...
                    if labels[i+1] == 1023: labels[i] = labels[i+2] - labels[i] # (769,770,771 ou 772) - 768 = 1,2,3 ou 4
                    else: labels[i] = labels[i+1] - labels[i] # a partir da proxima tarefa (769,770,771 ou 772) - 768 = 1,2,3 ou 4
            
        else: # if Evaluate dataset (A0sE.gdf) 
            labels = np.where(labels==5, 277, labels) # Idling EEG (eyes closed) 
            labels = np.where(labels==6, 276, labels) # Idling EEG (eyes open) 
            labels = np.where(labels==7, 32766, labels) # Start of a new run/segment (after a break)
            
            # muda padrão dos rotulos desconhecidos de 4 para 783 conforme descrição oficial do dataset
            idx4 = np.where(labels==4)
            labels[idx4] = trues + 768
            
            # rotula inicios de trials no dset de validação conforme rótulos verdadeiros fornecidos (truelabels)
            idx768 = np.where(labels==768)
            labels[idx768] = trues
    
    elif so in ['win','mac']:
        """ Dataset description MNE (MAC & Windows) (by vboas): more info in http://bbci.de/competition/iv/desc_2a.pdf
            Meta-info (Training data _T):
             	1=1023 (rejected trial)
             	2=1072 (Unknown/ Eye Moviments) 
             	3=276 (Eye open)                      [suj 4 = 32766 (Start a new run)]  
             	4=277 (Eye closed)                    [suj 4 = 768 (start trial)]
             	5=32766 (Start a new run)             [suj 4 = 769 (Class 1 - LH - cue onset)]
             	6=768 (start trial)                   [suj 4 = 770 (Class 2 - RH - cue onset)]
             	7=769 (Class 1 - LH - cue onset)      [suj 4 = 771 (Class 3 - Foot - cue onset)]
             	8=770 (Class 2 - RH - cue onset)      [suj 4 = 772 (Class 4 - Tongue - cue onset)]
             	9=771 (Class 3 - Foot - cue onset)    [suj 4 = None ] 
             	10=772 (Class 4 - Tongue - cue onset) [suj 4 = None ]
            Meta-info (Test data _E):
             	1=1023 (rejected trial)
             	2=1072 (Unknown/ Eye Moviments) 
             	3=276 (Eye open)
             	4=277 (Eye closed) 
             	5=32766 (Start a new run)
             	6=768 (start trial)
             	7=783 (Cue unknown/undefined)
        """
        if ds=='T': # if Training dataset (A0sT.gdf)
            labels = np.where(labels==1, 1023, labels) # Rejected trial
            labels = np.where(labels==2, 1072, labels) # Eye movements / Unknown 
            if suj == 4:
                labels = np.where(labels==3, 32766, labels) # Start of a new run/segment (after a break) 
                labels = np.where(labels==4, 768, labels) # Start trial t=0
                labels = np.where(labels==5, 769, labels) # LH (classe 1)
                labels = np.where(labels==6, 770, labels) # RH (classe 2) 
                labels = np.where(labels==7, 771, labels) # Foot (classe 3)
                labels = np.where(labels==8, 772, labels) # Tongue (classe 4)
            else:
                labels = np.where(labels==3, 276, labels) # Idling EEG (eyes open)
                labels = np.where(labels==4, 277, labels) # Idling EEG (eyes closed)
                labels = np.where(labels==5, 32766, labels) # Start of a new run/segment (after a break) 
                labels = np.where(labels==6, 768, labels) # Start trial t=0
                labels = np.where(labels==7, 769, labels) # LH (classe 1)
                labels = np.where(labels==8, 770, labels) # RH (classe 2) 
                labels = np.where(labels==9, 771, labels) # Foot (classe 3)
                labels = np.where(labels==10, 772, labels) # Tongue (classe 4)
            
            for i in range(0, len(labels)): 
                if labels[i]==768: # rotula [1 a 4] o inicio da trial...
                    if labels[i+1] == 1023: labels[i] = labels[i+2] - labels[i] # (769,770,771 ou 772) - 768 = 1,2,3 ou 4
                    else: labels[i] = labels[i+1] - labels[i] # a partir da proxima tarefa (769,770,771 ou 772) - 768 = 1,2,3 ou 4
            
        else: # if Evaluate dataset (A0sE.gdf)
            labels = np.where(labels==1, 1023, labels) # Rejected trial
            labels = np.where(labels==2, 1072, labels) # Eye movements / Unknown 
            labels = np.where(labels==3, 276, labels) # Idling EEG (eyes open) 
            labels = np.where(labels==4, 277, labels) # Idling EEG (eyes closed) 
            
            labels = np.where(labels==5, 32766, labels) # Start of a new run/segment (after a break)
            labels = np.where(labels==6, 768, labels) # Start trial t=0
            # labels = np.where(labels==7, 783, labels)
            
            # muda padrão dos rotulos desconhecidos de 7 para 783 conforme descrição oficial do dataset
            idx4 = np.where(labels==7)
            labels[idx4] = trues + 768
            
            # rotula inicios de trials no dset de validação conforme rótulos verdadeiros fornecidos (truelabels)
            idx768 = np.where(labels==768)
            labels[idx768] = trues
    
    return labels


def iv2b_labeling(ev, ds, suj, trues, so='lnx'): 
    # normaliza rotulos de eventos conforme descrição oficial do dataset
    if so == 'lnx':
        """ Dataset Description (by vboas): more info in http://bbci.de/competition/iv/desc_2b.pdf
            01T e 02T (without feedback)
        		10 trial * 2 classes * 6 runs * 2 sessions = 240 trials (120 per class)
        		Cross t=0 (per 3s)
        		beep t=2s
        		cue t=3s (per 1.25s)
        		MI t=4s (per 3s)
        		Pause t=7s (per 1.5-2.5s)
        		EndTrial t=8.5-9.5
        	03T, 04E e 05E (with feedback)
        		10 trial * 2 classes * 4 runs * 3 sessions = 240 trials (120 per class)
        		Smiley(grey) t=0 (per 3.5s)
        		beep t=2s
        		cue t=3s (per 4.5s)
        		MI (Feedback períod) t=3.5s (per 4s)
        		Pause t=7.5s (per 1-2s)
        		EndTrial t=8.5-9.5
        	Meta-info 01T e 02T:
        		1=1023 (rejected trial)
        		2=768 (start trial)
        		3=769 (Class 1 - LH - cue onset)
        		4=770 (Class 2 - RH - cue onset)
        		5=277 (Eye closed)
        		6=276 (Eye open)
        		7=1081 (Eye blinks)
        		8=1078 (Eye rotation)
        		9=1077 (Horizontal eye movement)
        		10=32766 (Start a new run) *(to B0102T == 5)
        		11=1078 (Vertical eye movement)			
        	Meta-info 03T:
        		1=781 (BCI feedback - continuous)
        		2=1023 (rejected trial)
        		3=768 (start trial)
        		4=769 (Class 1 - LH - cue onset)
        		5=770 (Class 2 - RH - cue onset)
        		6=277 (Eye closed)
        		7=276 (Eye open)
        		8=1081 (Eye blinks)
        		9=1078 (Eye rotation)
        		10=1077 (Horizontal eye movement)
        		11=32766 (Start a new run)
        		12=1078 (Vertical eye movement)
        	Meta-info 04E e 05E:
        		1=781 (BCI feedback - continuous)
        		2=1023 (rejected trial)
        		3=768 (start trial)
        		4=783 (Cue unknown/undefined)
        		5=277 (Eye closed)
        		6=276 (Eye open)
        		7=1081 (Eye blinks)
        		8=1078 (Eye rotation)
        		9=1077 (Horizontal eye movement)
        		10=32766 (Start a new run)
        		11=1078 (Vertical eye movement)
        """
        # Remove marcações inúteis e normaliza rotulos de eventos conforme descrição oficial do dataset
        if ds in ['01T','02T']:
            for rm in range(5,12): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
            ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete rejected trials
            ev[:,1] = np.where(ev[:,1]==2, 0, ev[:,1]) # altera label start trial de 2 para 0
            ev[:,1] = np.where(ev[:,1]==3, 1, ev[:,1]) # altera label cue LH de 3 para 1
            ev[:,1] = np.where(ev[:,1]==4, 2, ev[:,1]) # altera label cue RH de 4 para 2
            for i in range(len(ev[:,1])):
                if ev[i,1]==0: ev[i,1] = ev[i+1,1] + 768
        elif ds=='03T': 
            for rm in range(6,13): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
            ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # delete rejected trials
            ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete feedback continuous
            ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1]) # altera label start trial de 3 para 0
            ev[:,1] = np.where(ev[:,1]==4, 1, ev[:,1]) # altera label cue LH de 4 para 1
            ev[:,1] = np.where(ev[:,1]==5, 2, ev[:,1]) # altera label cue RH de 5 para 2
            for i in range(len(ev[:,1])):
                if ev[i,1]==0: ev[i,1] = ev[i+1,1] + 768
        else:
            for rm in range(5,12): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
            ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # delete rejected trials
            ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete feedback continuous
            ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1]) # altera label start trial de 3 para 0
            ev[np.where(ev[:,1]==4),1] = trues #rotula momento da dica conforme truelabels
            for i in range(len(ev[:,1])):
                if ev[i,1]==0: ev[i,1] = ev[i+1,1] + 768
    
    elif so in ['win','mac']:
        pass
    
    return ev


def twl_labeling(ev, ds): # normaliza rotulos de eventos conforme descrição oficial do dataset
    """ Dataset description/Meta-info MNE (Linux) (by vboas):
        1=Cross on screen (BCI experiment) 
        2=Feedback (continuous) - onset (BCI experiment)
        3=768 Start of Trial, Trigger at t=0s
        4=783 Unknown
        5=769 class1, Left hand - cue onset (BCI experiment)
        6=770 class2, Right hand - cue onset (BCI experiment)
    """
    ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # elimina marcações inuteis (cross on screen)
    ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # elimina marcações inuteis (feedback continuous)
    ev = np.delete(ev,np.where(ev[:,1]==4),axis=0) # elimina marcações inuteis (unknown)
    ev[:,1] = np.where(ev[:,1]==5, 769, ev[:,1]) # altera label lh de 5 para 1
    ev[:,1] = np.where(ev[:,1]==6, 770, ev[:,1]) # altera label rh de 6 para 2
    
    for i in range(len(ev)): 
        if ev[i,1]==3: ev[i,1] = ev[i+1,1] - 768 # rotula dica conforme tarefa (idx+1=769 ou 770, idx=1 ou 2)

    return ev


def iii3a_save_npy(suj, path, channels=60):
    mne.set_log_level('WARNING','DEBUG')
    raw = mne.io.read_raw_gdf(path + '/gdf/' + suj + '.gdf')
    raw.load_data()
    data = raw.get_data() # [channels x samples]
    # data = data[:channels]
    # data = corrigeNaN(data) # Correção de NaN nos dados brutos
    events_raw = raw.find_edf_events()
    events = np.delete(events_raw[0],1,axis=1) # elimina coluna de zeros
    truelabels = np.ravel(pd.read_csv(path + '/gdf/true_labels/trues_' + suj + '.csv'))
    events = iii3a_labeling(events, suj, truelabels) # Labeling correctly the events like competition description
    np.save(path + '/npy/' + suj + '_data', data)
    np.save(path + '/npy/' + suj + '_events', events)


def iii4a_save_npy(suj, path):
    mat = loadmat(path + '/mat/' + suj + '.mat')
    data = mat['cnt'].T # 0.1 * mat['cnt'].T # convert to uV
    pos = mat['mrk'][0][0][0][0]
    true_mat = loadmat(path + '/mat/true_labels/trues_' + suj + '.mat')
    true_y = np.ravel(true_mat['true_y']) # RH=1 Foot=2
    true_y = np.where(true_y == 2, 3, true_y) # RH=1 Foot=3
    # true_test_idx = np.ravel(true_mat['test_idx'])
    events = np.c_[pos, true_y]
    # data = corrigeNaN(data)
    # data = np.asarray([ np.nan_to_num(dt) for dt in data ])
    # data = np.asarray([ np.ravel(pd.DataFrame(dt).fillna(pd.DataFrame(dt).mean())) for dt in data ])        
    np.save(path + '/npy/' + suj + '_data', data)
    np.save(path + '/npy/' + suj + '_events', events)  


def iv2a_save_npy(suj, path, channels=25):
    mne.set_log_level('WARNING','DEBUG')
    dataT, dataV, eventsT, eventsV = [],[],[],[]
    for ds in ['T', 'E']:
        raw = mne.io.read_raw_gdf(path + '/gdf/A0' + str(suj) + ds + '.gdf')
        raw.load_data()
        data = raw.get_data() # [channels x samples]
        # data = data[:channels]
        # data = self.corrigeNaN(data) # Correção de NaN nos dados brutos
        raw_events = raw.find_edf_events()
        events = np.delete(raw_events[0], 1, axis=1) # elimina coluna de zeros do MNE
        truelabels = np.ravel(loadmat(path + '/gdf/true_labels/A0' + str(suj) + 'E.mat' )['classlabel']) # Loading true labels to use in evaluate files (E)
        events[:,1] = iv2a_labeling(events[:,1], ds, suj, truelabels, 'lnx') # Labeling correctly the events like competition description ('win','mac','lnx')
        if ds == 'T': dataT, eventsT = data, events
        else: dataV, eventsV = data, events

    # Save separated npy files to each dataset type (train and train)
    # np.save(path_to_npy + str(suj) + '_dataT', dataT)
    # np.save(path_to_npy + str(suj) + '_dataV', dataV)
    # np.save(path_to_npy + str(suj) + '_eventsT', eventsT)
    # np.save(path_to_npy + str(suj) + '_eventsV', eventsV)
    
    all_data = np.c_[dataT, dataV]    
    new_events = np.copy(eventsV)
    new_events[:,0] += len(dataT.T) # eventsV pos + last dataT pos (eventsT is continued by eventsV)
    all_events = np.r_[eventsT, new_events]
    
    # Save a unique npy file with both dataset type (train + test)
    np.save(path + '/npy/A0' + str(suj) + '_data', all_data)    
    np.save(path + '/npy/A0' + str(suj) + '_events', all_events)


def iv2b_save_npy(suj, path, channels=3):
    mne.set_log_level('WARNING','DEBUG')
    DT, EV = [], []
    for ds in ['01T','02T','03T','04E','05E']:
        raw = mne.io.read_raw_gdf(path + '/gdf/B0' + str(suj) + ds + '.gdf')
        raw.load_data()
        data = raw.get_data() # [channels x samples]
        # data = data[:channels]
        # data = corrigeNaN(data) # Correção de NaN nos dados brutos
        events = raw.find_edf_events()
        events = np.delete(events[0],1,axis=1) # elimina coluna de zeros
        truelabels = np.ravel(loadmat(path + '/gdf/true_labels/B0' + str(suj) + ds + '.mat')['classlabel'])
        events = iv2b_labeling(events, ds, suj, truelabels, 'lnx') # Labeling correctly the events like competition description
        
        # Save separated npy files to each dataset
        # np.save(path + '/npy/B0' + str(suj) + ds + '_data', data)
        # np.save(path + '/npy/B0' + str(suj) + ds + '_events', events)
        
        DT.append(data)
        EV.append(events)
        
    # Save a unique npy file with all datasets
    soma = 0
    for i in range(1,len(EV)): 
        soma += len(DT[i-1].T)
        EV[i][:,0] += soma
        
    all_data = np.c_[DT[0],DT[1],DT[2],DT[3],DT[4]]
    all_events = np.r_[EV[0],EV[1],EV[2],EV[3],EV[4]]
    
    np.save(path + '/npy/B0' + str(suj) + '_data', all_data)    
    np.save(path + '/npy/B0' + str(suj) + '_events', all_events)


def cl_save_npy(suj, path):
    data = np.load(path + '/original/orig_' + suj + '_data.npy').T
    events = np.load(path + '/original/orig_' + suj + '_events.npy').astype(int)
    # data = corrigeNaN(data)
    for i in range(len(events)-1): 
        if events[i,1]==0:
            events[i,1] = events[i+1,1]
            events[i+1,1] = events[i+1,1] + 768
    # if suj=='CL_LF': events[:,1] = np.where(events[:,1] == 2, 3, events[:,1]) # LH=1, FooT=3
    np.save(path + '/' + suj + '_data.npy', data)
    np.save(path + '/' + suj + '_events.npy', events)


def twl_save_npy(suj, path, channels=8):
    mne.set_log_level('WARNING','DEBUG')
    dataT, dataV, eventsT, eventsV = [],[],[],[]
    for ds in ['S1','S2']:
        raw = mne.io.read_raw_gdf(path + '/gdf/' + suj + '_' + ds + '.gdf')
        raw.load_data()
        data = raw.get_data() # [channels x samples]
        # data = data[:channels]
        # data = corrigeNaN(data) # Correção de NaN nos dados brutos
        events_raw = raw.find_edf_events()
        events = np.delete(events_raw[0],1,axis=1) # elimina coluna de zeros
        events = twl_labeling(events, ds) # Labeling correctly the events like competition description
        
        # Save separated npy files to each dataset type (train and train)
        # np.save(path + '/npy/' + suj + '_' + ds + '_data', data)
        # np.save(path + '/npy/' + suj + '_' + ds + '_events', events)
        
        if ds == 'S1': dataT, eventsT = data, events
        else: dataV, eventsV = data, events
    
    all_data = np.c_[dataT, dataV]    
    new_events = np.copy(eventsV)
    new_events[:,0] += len(dataT.T) # eventsV pos + last dataT pos (eventsT is continued by eventsV)
    all_events = np.r_[eventsT, new_events]
    
    np.save(path + '/npy/' + suj + '_data', all_data)    
    np.save(path + '/npy/' + suj + '_events', all_events)


def extractEpochs(data, events, smin, smax, class_ids):
    events_list = events[:, 1] # get class labels column
    cond = False
    for i in range(len(class_ids)): cond += (events_list == class_ids[i]) #get only class_ids pos in events_list
    idx = np.where(cond)[0]
    s0 = events[idx, 0] # get initial timestamps of each class epochs
    sBegin = s0 + smin
    sEnd = s0 + smax
    n_epochs = len(sBegin)
    n_channels = data.shape[0]
    n_samples = smax - smin
    epochs = np.zeros([n_epochs, n_channels, n_samples])
    labels = events_list[idx]
    bad_epoch_list = []
    for i in range(n_epochs):
        epoch = data[:, sBegin[i]:sEnd[i]]
        if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch # Check if epoch is complete
        else:
            print('Incomplete epoch detected...')
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    return epochs, labels
    
    
def nanCleaner(epoch):
    """Removes NaN from data by interpolation
    data_in : input data - np matrix channels x samples
    data_out : clean dataset with no NaN samples"""
    for i in range(epoch.shape[0]):
        bad_idx = np.isnan(epoch[i, :])
        epoch[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoch[i, ~bad_idx])
    return epoch
    
    
def corrigeNaN(data):
    for ch in range(data.shape[0] - 1):
        this_chan = data[ch]
        data[ch] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(data[ch])
        meanChannel = np.nanmean(data[ch])
        data[ch, mask] = meanChannel
    return data

   
class Filter:
    def __init__(self, fl, fh, buffer_len, srate, filt_info, forder=None, band_type='bandpass'):
        self.ftype = filt_info['design']
        if fl == 0: fl = 0.001
        self.nyq = 0.5 * srate
        low = fl / self.nyq
        high = fh / self.nyq
        self.res_freq = (srate / buffer_len)
        if high >= 1: high = 0.99

        if self.ftype == 'IIR':
            self.forder = filt_info['iir_order']
            #self.b, self.a = iirfilter(forder, [low, high], btype=band_type)
            self.b, self.a = butter(self.forder, [low, high], btype=band_type)
        elif self.ftype == 'FIR':
            self.forder = filt_info['fir_order']
            self.b = firwin(self.forder, [low, high], window='hamming', pass_zero=False)
            self.a = [1]
        elif self.ftype == 'DFT':
            self.bmin = int(fl / self.res_freq)  # int(fl * (srate/self.nyq)) # int(low * srate)
            self.bmax = int(fh / self.res_freq)  # int(fh * (srate/self.nyq)) # int(high * srate)


    def apply_filter(self, data_in, is_epoch=False):
        if self.ftype != 'DFT':
            data_out = filtfilt(self.b, self.a, data_in)
            data_out = lfilter(self.b, self.a, data_in)
        else:
            if is_epoch:
                data_out = fft(data_in)
                REAL = np.real(data_out)[:, self.bmin:self.bmax].T
                IMAG = np.imag(data_out)[:, self.bmin:self.bmax].T
                data_out = np.transpose(list(itertools.chain.from_iterable(zip(IMAG, REAL))))
            else:
                data_out = fft(data_in)
                REAL = np.transpose(np.real(data_out)[:, :, self.bmin:self.bmax], (2, 0, 1))
                IMAG = np.transpose(np.imag(data_out)[:, :, self.bmin:self.bmax], (2, 0, 1))
                data_out = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
                data_out = np.transpose(data_out, (1, 2, 0))
        return data_out


class CSP():
    def __init__(self, n_components):
        self.n_components = n_components
        self.filters_ = None
    def fit(self, X, y):
        e, c, s = X.shape
        classes = np.unique(y)   
        Xa = X[classes[0] == y,:,:]
        Xb = X[classes[1] == y,:,:]
        S0 = np.zeros((c, c)) 
        S1 = np.zeros((c, c))
        for epoca in range(int(e/2)):
            # S0 = np.add(S0, np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T), out=S0, casting="unsafe")
            # S1 = np.add(S1, np.dot(Xb[epoca,:,:], Xb[epoca,:,:].T), out=S1, casting="unsafe")
            S0 += np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T) #covA Xa[epoca]
            S1 += np.dot(Xb[epoca,:,:], Xb[epoca,:,:].T) #covB Xb[epoca]
        [D, W] = eigh(S0, S0 + S1)
        ind = np.empty(c, dtype=int)
        ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
        ind[1::2] = np.arange(0, c // 2)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        return self # instruction add because cross-validation pipeline
    def transform(self, X):        
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        XVAR = np.log(np.mean(XT ** 2, axis=2)) # Xcsp
        return XVAR


class BCI():
    
    def __init__(self, data, events, class_ids, overlap, fs, crossval, nfolds, test_perc,
                 f_low=None, f_high=None, tmin=None, tmax=None, ncomp=None, ap=None, filt_info=None, clf=None):
        self.data = data
        self.events = events
        self.class_ids = class_ids
        self.overlap = overlap
        self.fs = fs
        self.crossval = crossval
        self.nfolds = nfolds
        self.test_perc = test_perc
        
        self.f_low = f_low
        self.f_high = f_high
        self.tmin = tmin
        self.tmax = tmax
        self.ncomp = ncomp
        self.ap = ap
        self.filt_info = filt_info
        self.clf = clf
     
        # print(self.class_ids, self.fs, self.crossval, self.f_low, self.f_high, self.tmin, self.tmax, self.ncomp,
        #       self.ap, self.filt_info, self.clf)
        
    def objective(self, args):
        print('\n', args)
        self.f_low, self.f_high, self.tmin, self.tmax, self.ncomp, self.ap, self.filt_info, self.clf = args
        self.f_low, self.f_high, self.ncomp = int(self.f_low), int(self.f_high), int(self.ncomp)

        while (self.tmax-self.tmin)<1: self.tmax+=0.5 # garante janela minima de 1seg
        # sleep(1)
        return self.evaluate() * (-1)    
    
    
    def evaluate(self):
        
        if self.clf['model'] == 'LDA': 
            # if clf_dict['lda_solver'] == 'svd': lda_shrinkage = None
            # else:
            #     lda_shrinkage = self.clf['shrinkage'] if self.clf['shrinkage'] in [None,'auto'] else self.clf['shrinkage']['shrinkage_float']
            self.clf_final = LDA(solver=self.clf['lda_solver'], shrinkage=None)
        
        if self.clf['model'] == 'Bayes': self.clf_final = GaussianNB()
        
        if self.clf['model'] == 'SVM': 
            # degree = self.clf['kernel']['degree'] if self.clf['kernel']['kf'] == 'poly' else 3
            # gamma = self.clf['gamma'] if self.clf['gamma'] in ['scale', 'auto'] else 10 ** (self.clf['gamma']['gamma_float'])
            self.clf_final = SVC(kernel=self.clf['kernel']['kf'], C=10 ** (self.clf['C']), 
                                 gamma='scale', degree=3, probability=True)
        
        if self.clf['model'] == 'KNN':   
            self.clf_final = KNeighborsClassifier(n_neighbors=int(self.clf['neig']), 
                                                  metric=self.clf['metric'], p=3) # p=self.clf['p'] #metric='minkowski', p=3)  # minkowski,p=2 -> distancia euclidiana padrão
                                                  
        if self.clf['model'] == 'DTree': 
            # print(self.clf['min_split'])
            # if self.clf['min_split'] == 1.0: self.clf['min_split'] += 1
            # max_depth = self.clf['max_depth'] if self.clf['max_depth'] is None else int(self.clf['max_depth']['max_depth_int'])
            
            self.clf_final = DecisionTreeClassifier(criterion=self.clf['crit'], random_state=0,
                                                    max_depth=None, # max_depth=max_depth,
                                                    min_samples_split=2 # min_samples_split=self.clf['min_split'], #math.ceil(self.clf['min_split']),
                                                    ) # None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
        
        
        if self.clf['model'] == 'MLP':   
            self.clf_final = MLPClassifier(verbose=False, max_iter=1000, tol=0.0001,
                                           learning_rate_init=10**self.clf['eta'],
                                           alpha=10**self.clf['alpha'],
                                           activation=self.clf['activ']['af'],
                                           hidden_layer_sizes=(int(self.clf['n_neurons']), int(self.clf['n_hidden'])),
                                           learning_rate='constant', # self.clf['eta_type'], 
                                           solver=self.clf['mlp_solver'],
                                           )
        
        smin = math.floor(self.tmin * self.fs)
        smax = math.floor(self.tmax * self.fs)
        self.buffer_len = smax - smin
        
        self.epochs, self.labels = extractEpochs(self.data, self.events, smin, smax, self.class_ids)
        self.epochs = nanCleaner(self.epochs)
        # self.epochs = np.asarray([ nanCleaner(ep) for ep in self.epochs ])
        
        self.filt = Filter(self.f_low, self.f_high, self.buffer_len, self.fs, self.filt_info)
        
        self.csp = CSP(n_components=self.ncomp)
        
        if self.crossval:
            kf = StratifiedShuffleSplit(self.nfolds, test_size=self.test_perc, random_state=42)
            #kf = StratifiedKFold(self.nfolds, False)
            
            if self.ap['option'] == 'classic':
                self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
                XF = self.filt.apply_filter(self.epochs)
                cross_scores = cross_val_score(self.chain, XF, self.labels, cv=kf)
                
                # cross_scores = []
                # for idx_treino, idx_teste in kf.split(self.epochs, self.labels):
                #     XT, XV, yT, yV = self.epochs[idx_treino], self.epochs[idx_teste], self.labels[idx_treino], self.labels[idx_teste]
                #     self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
                #     self.chain.fit(XT, yT)
                #     self.csp_filters = self.chain['CSP'].filters_
                #     cross_scores.append(self.chain.score(XV, yV))

                self.acc = np.mean(cross_scores)
            
            elif self.ap['option'] == 'sbcsp':
                cross_scores = []
                for idx_treino, idx_teste in kf.split(self.epochs, self.labels):
                    cross_scores.append(self.sbcsp_approach(self.epochs[idx_treino], self.epochs[idx_teste], 
                                                            self.labels[idx_treino], self.labels[idx_teste]))
                self.acc = np.mean(cross_scores)   
                
        else:
            
            test_perc = 0.5
            test_size = int(len(self.epochs) * test_perc)
            train_size = int(len(self.epochs) - test_size)
            train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
            epochsT, labelsT = self.epochs[:train_size], self.labels[:train_size] 
            epochsV, labelsV = self.epochs[train_size:], self.labels[train_size:]
            
            XT = [ epochsT[np.where(labelsT == i)] for i in self.class_ids ] # Extrair épocas de cada classe
            XV = [ epochsV[np.where(labelsV == i)] for i in self.class_ids ]
            
            XT = np.concatenate([XT[0],XT[1]]) # Train data classes A + B
            XV = np.concatenate([XV[0],XV[1]]) # Test data classes A + B        
            yT = np.concatenate([self.class_ids[0] * np.ones(int(len(XT)/2)), self.class_ids[1] * np.ones(int(len(XT)/2))])
            yV = np.concatenate([self.class_ids[0] * np.ones(int(len(XV)/2)), self.class_ids[1] * np.ones(int(len(XV)/2))])
            # print(XT.shape, XV.shape)
            if self.ap['option'] == 'classic':
                self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
                XTF = self.filt.apply_filter(XT)
                XVF = self.filt.apply_filter(XV)
                
                # csp.fit(XTF, yT)
                # XT_CSP = csp.transform(XTF)
                # XV_CSP = csp.transform(XVF) 
                # svc_final.fit(XT_CSP, yT)
                # scores = svc_final.predict(XV_CSP)
                # acc = np.mean(scores == yV)
                
                self.chain.fit(XTF, yT)
                self.csp_filters = self.chain['CSP'].filters_
                self.acc = self.chain.score(XVF, yV)
            
            elif self.ap['option'] == 'sbcsp':
                self.acc = self.sbcsp_approach(XT, XV, yT, yV)
        
        return self.acc
    
    
    def sbcsp_approach(self, XT, XV, yT, yV):
        self.chain = Pipeline([('CSP', CSP(n_components=self.ncomp)), ('LDA', LDA()), ('SVC', self.clf_final)])
        nbands = int(self.ap['nbands'])
        nbands = (self.f_high-self.f_low)-1 if nbands > (self.f_high-self.f_low) else nbands
        
        if self.filt_info['design'] == 'DFT':
            XT_FFT = self.filt.apply_filter(XT)
            XV_FFT = self.filt.apply_filter(XV)
            n_bins = len(XT_FFT[0, 0, :])  # ou (fh-fl) * 4 # Número total de bins de frequencia
        else: n_bins = self.f_high - self.f_low
        overlap = 0.5 if self.overlap else 1
        step = int(n_bins / nbands)
        size = int(step / overlap) 
        SCORE_T = np.zeros((len(XT), nbands))
        SCORE_V = np.zeros((len(XV), nbands))
        self.csp_filters_sblist = []
        self.lda_sblist = []
        for i in range(nbands):
            if self.filt_info['design'] == 'DFT':
                bin_ini = i * step
                bin_fim = i * step + size
                if bin_fim >= n_bins: bin_fim = n_bins - 1
                XTF = XT_FFT[:, :, bin_ini:bin_fim]
                XVF = XV_FFT[:, :, bin_ini:bin_fim]
            else:
                fl_sb = i * step + self.f_low
                fh_sb = i * step + size + self.f_low
                if fh_sb > self.f_high: fh_sb = self.f_high
                filt_sb = Filter(fl_sb, fh_sb, len(XT[0,0,:]), self.fs, self.filt_info)
                XTF = filt_sb.apply_filter(XT)
                XVF = filt_sb.apply_filter(XV)
                # print(fl_sb, fh_sb)
                
            self.chain['CSP'].fit(XTF, yT)
            XT_CSP = self.chain['CSP'].transform(XTF)
            XV_CSP = self.chain['CSP'].transform(XVF)
            self.chain['LDA'].fit(XT_CSP, yT)
            SCORE_T[:, i] = np.ravel(self.chain['LDA'].transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
            SCORE_V[:, i] = np.ravel(self.chain['LDA'].transform(XV_CSP))
            self.csp_filters_sblist.append(self.chain['CSP'].filters_)
            self.lda_sblist.append(self.chain['LDA'])
            
            # csp = CSP(n_components=ncomp)
            # lda = LDA()
            # csp.fit(XTF, yT)
            # XT_CSP = csp.transform(XTF)
            # XV_CSP = csp.transform(XVF)
            
            # lda.fit(XT_CSP, yT)
            # SCORE_T[:, i] = np.ravel(lda.transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
            # SCORE_V[:, i] = np.ravel(lda.transform(XV_CSP))
    
        SCORE_T0 = SCORE_T[yT == self.class_ids[0], :]
        SCORE_T1 = SCORE_T[yT == self.class_ids[1], :]
        self.p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
        self.p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
        META_SCORE_T = np.log(self.p0.pdf(SCORE_T) / self.p1.pdf(SCORE_T))
        META_SCORE_V = np.log(self.p0.pdf(SCORE_V) / self.p1.pdf(SCORE_V))
    
        self.chain['SVC'].fit(META_SCORE_T, yT)
        self.scores = self.chain['SVC'].predict(META_SCORE_V)
        
        # clf_final.fit(META_SCORE_T, yT)
        # scores = clf_final.predict(META_SCORE_V)

        return np.mean(self.scores == yV)


    

    