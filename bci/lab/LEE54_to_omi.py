import pickle
import numpy as np
from scipy.io import loadmat
from datetime import datetime

"""
 'EEG_MI_train' and 'EEG_MI_test': training and test data
 'x':       continuous EEG signals (data points × channels)
 't':       stimulus onset times of each trial
 'fs':      sampling rates
 'y_dec':   class labels in integer types 
 'y_logic': class labels in logical types
 'y_class': class definitions
 'chan':    channel information

 Protocol: Tcue=3s, Tpause=7s, mi_time=4s, min_pause_time=6s, endTrial=13~14.5s
 100 trials (50 LH + 50 RH) * 2 phases (train, test) * 2 sessions = 400 trials (200 LH + 200 RH)
 62 channels
 54 subjects
 fs = 1000 Hz
"""
fs = 1000
class_ids = [1, 2]
path = '/mnt/dados/eeg_data/LEE54/'
subjects = range(1, 55)
ch_labels = list(['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4',
                  'T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz',
                  'O2','PO10','FC3','FC4','C5','C1','C2','C6','CP3','CPz','CP4','P1','P2','POz','FT9',
                  'FTT9h','TTP7h','TP7','TPP9h','FT10','FTT10h','TPP8h','TP8','TPP10h','F9','F10','AF7',
                  'AF3','AF4','AF8','PO3','PO4'])

for suj in subjects:
    suj_in = str(suj) if suj>=10 else ('0' + str(suj))
    
    mat1 = loadmat(path + 'session1/sess01_subj' + suj_in + '_EEG_MI.mat')
    mat2 = loadmat(path + 'session2/sess02_subj' + suj_in + '_EEG_MI.mat')
    
    TRAIN1 = mat1['EEG_MI_test']
    TEST1 = mat1['EEG_MI_train']
    TRAIN2 = mat2['EEG_MI_test']
    TEST2 = mat2['EEG_MI_train']
    
    dataT1 = TRAIN1['x'][0,0].T
    dataV1 = TEST1['x'][0,0].T
    dataT2 = TRAIN2['x'][0,0].T
    dataV2 = TEST2['x'][0,0].T
    
    eventsT1 = np.r_[ TRAIN1['t'][0,0], TRAIN1['y_dec'][0,0] ].T
    eventsV1 = np.r_[ TEST1['t'][0,0], TEST1['y_dec'][0,0] ].T
    eventsT2 = np.r_[ TRAIN2['t'][0,0], TRAIN2['y_dec'][0,0] ].T
    eventsV2 = np.r_[ TEST2['t'][0,0], TEST2['y_dec'][0,0] ].T
    
    
    eventsT1[:, 1] = np.where(eventsT1[:, 1]==2, 1, 2) # troca class_ids 1=LH, 2=RH
    eventsV1[:, 1] = np.where(eventsV1[:, 1]==2, 1, 2)
    eventsT2[:, 1] = np.where(eventsT2[:, 1]==2, 1, 2) # troca class_ids 1=LH, 2=RH
    eventsV2[:, 1] = np.where(eventsV2[:, 1]==2, 1, 2)
        
    
    all_data = np.c_[dataT1, dataT2, dataV1, dataV2]  
    
    # opção 1
    eventsT2[:,0] += dataT1.shape[-1] # eventsT2 pos + last dataT1 pos (eventsT2 is continued by eventsT1) ...
    eventsV1[:,0] += (dataT1.shape[-1] + dataT2.shape[-1])
    eventsV2[:,0] += (dataT1.shape[-1] + dataT2.shape[-1] + dataV1.shape[-1])
    all_events = np.r_[eventsT1, eventsT2, eventsV1, eventsV2]
     
    # opção 2
    # DT = [dataT1, dataT2, dataV1, dataV2]
    # EV = [eventsT1, eventsT2, eventsV1, eventsV2]
    # soma = 0
    # for i in range(1,len(EV)): 
    #     soma += len(DT[i-1].T)
    #     EV[i][:,0] += soma
    # all_events = np.r_[EV[0],EV[1],EV[2],EV[3]]
    
    
    info = {'fs': fs, 'class_ids': class_ids, 'trial_tcue': 3.0,
            'trial_tpause': 7.0, 'trial_mi_time': 4.0, 'trials_per_class': 200,
            'eeg_channels': 62, 'ch_labels': ch_labels,
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
    
    
    omi_data = [ all_data, all_events, info ]
    with open(path + '/omi/subj' + suj + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)

    
    epochsT1 = TRAIN1['smt'][0,0]
    epochsV1 = TEST1['smt'][0,0]
    epochsT2 = TRAIN2['smt'][0,0]
    epochsV2 = TEST2['smt'][0,0]
    
    

    # fs = TRAIN['fs'][0,0][0,0]
    # chan = np.asarray([ TRAIN['chan'][0,0][0,i][0] for i in range(TRAIN['chan'][0,0].shape[-1]) ])
    # class_names = np.asarray(list(TRAIN['class'][0,0][:, 1])).reshape(-1)
    # class_ids = np.asarray(list(TRAIN['class'][0,0][:, 0])).reshape(-1)
    # print(class_names, class_ids)
    
    # pos_T = TRAIN['t'][0,0]
    # pos_V = TEST['t'][0,0]
    # ev_ids_T = TRAIN['y_dec'][0,0]
    # ev_ids_V = TEST['y_dec'][0,0]
    # y_logic = TRAIN['y_logic'][0,0]
    # y_class = np.asarray([ TRAIN['y_class'][0,0][0,i][0] for i in range(TRAIN['y_class'][0,0].shape[-1]) ])

        
        
# data, events, info = pickle.load(open(path + '/omi/subj' + suj + '.omi', 'rb'))


