# coding: utf-8
import os
import mne
import pandas as pd
import warnings
import numpy as np
from scipy.io import loadmat, savemat
from datetime import datetime
from scipy.signal import decimate, resample
# import pickle
# from proc.utils import save_npy_data, load_npy_data, PATH_TO_SESSION, extractEpochs, nanCleaner

warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

# path = '/mnt/dados/eeg_data/IV2a/'  ## >>> ENTER THE PATH TO THE DATASET HERE
# # path_out = path + 'npy/'
# # if not os.path.isdir(path_out): os.makedirs(path_out)

def iii3a(path, subj, ds):
    raw = mne.io.read_raw_gdf(path).load_data()
    d = raw.get_data()[:60]  # [channels x samples]
    # d = corrigeNaN(d)
    e_raw = mne.events_from_annotations(raw)  # raw.find_edf_events()
    e = np.delete(e_raw[0], 1, axis=1)  # elimina coluna de zeros
    truelabels = np.ravel(pd.read_csv(os.path.dirname(path) + '/true_labels/' + subj + '.csv'))
    ch_names = raw.ch_names

    cond = False
    for i in [1, 2, 3]: cond += (e[:, 1] == i)
    idx = np.where(cond)[0]
    e = np.delete(e, idx, axis=0)

    e[:, 1] = np.where(e[:, 1] == 4, 0, e[:, 1])  # Labeling Start trial t=0

    idx = np.where(e[:, 1] != 0)
    e[idx, 1] = truelabels

    for i in range(0, len(e)):
        if e[i, 1] == 0: e[i, 1] = (e[i + 1, 1] + 10)  # labeling start trial [11 a 14] according cue [1,2,3,4]

    info = {'fs': 250, 'class_ids': [1, 2, 3, 4], 'trial_tcue': 3.0, 'trial_tpause': 7.0, 'trial_mi_time': 4.0,
            'trials_per_class': 90 if subj == 'K3' else 60, 'eeg_channels': d.shape[0], 'ch_labels': ch_names,
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}

    return [d, e, info]


def iv2a(path, subj, ds):

    if 'T.gdf' in path:
        # raw = mne.io.read_raw_gdf(path + 'A0' + str(suj) + 'T.gdf').load_data()
        raw = mne.io.read_raw_gdf(path).load_data()
        dt = raw.get_data()[:22]  # [channels x samples]
        et_raw = mne.events_from_annotations(raw)  # raw.find_edf_events()

        et = np.delete(et_raw[0], 1, axis=1)  # remove MNE zero columns
        et = np.delete(et, np.where(et[:, 1] == 1), axis=0)  # remove rejected trial
        et = np.delete(et, np.where(et[:, 1] == 3), axis=0)  # remove eye movements/unknown
        et = np.delete(et, np.where(et[:, 1] == 8), axis=0)  # remove eyes closed
        et = np.delete(et, np.where(et[:, 1] == 9), axis=0)  # remove eyes open
        et = np.delete(et, np.where(et[:, 1] == 10), axis=0)  # remove start of a new run/segment
        et[:, 1] = np.where(et[:, 1] == 2, 0, et[:, 1])  # start trial t=0
        et[:, 1] = np.where(et[:, 1] == 4, 1, et[:, 1])  # LH
        et[:, 1] = np.where(et[:, 1] == 5, 2, et[:, 1])  # RH
        et[:, 1] = np.where(et[:, 1] == 6, 3, et[:, 1])  # Foot
        et[:, 1] = np.where(et[:, 1] == 7, 4, et[:, 1])  # Tongue
        for i in range(0, len(et)):
            if et[i, 1] == 0: et[i, 1] = (et[i + 1, 1] + 10)  # labeling start trial [11 a 14] according cue [1,2,3,4]

        info = {'fs': 250, 'class_ids': [1, 2, 3, 4], 'trial_tcue': 2.0, 'trial_tpause': 6.0, 'trial_mi_time': 4.0,
                'trials_per_class': 72, 'eeg_channels': dt.shape[0], 'ch_labels': raw.ch_names,
                'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}

        # np.save(os.path.dirname(path) + '/A0' + str(subj) + 'T', [dt, et, info], allow_pickle=True)

        return [dt, et, info]

    elif 'E.gdf' in path:
        raw = mne.io.read_raw_gdf(path).load_data()
        trues = np.ravel(loadmat(os.path.dirname(path) + '/true_labels/A0' + str(subj) + 'E.mat')['classlabel'])
        dv = raw.get_data()[:22]  # [channels x samples]
        ev_raw = mne.events_from_annotations(raw)  # raw.find_edf_events()
        ev = np.delete(ev_raw[0], 1, axis=1)  # remove MNE zero columns
        ev = np.delete(ev, np.where(ev[:, 1] == 1), axis=0)  # remove rejected trial
        ev = np.delete(ev, np.where(ev[:, 1] == 3), axis=0)  # remove eye movements/unknown
        ev = np.delete(ev, np.where(ev[:, 1] == 5), axis=0)  # remove eyes closed
        ev = np.delete(ev, np.where(ev[:, 1] == 6), axis=0)  # remove eyes open
        ev = np.delete(ev, np.where(ev[:, 1] == 7), axis=0)  # remove start of a new run/segment
        ev[:, 1] = np.where(ev[:, 1] == 2, 0, ev[:, 1])  # start trial t=0
        ev[np.where(ev[:, 1] == 4), 1] = trues  # change unknown value labels(4) to value in [1,2,3,4]
        for i in range(0, len(ev)):
            if ev[i, 1] == 0: ev[i, 1] = (ev[i + 1, 1] + 10)  # labeling start trial [11 a 14] according cue [1,2,3,4]

        info = {'fs': 250, 'class_ids': [1, 2, 3, 4], 'trial_tcue': 2.0, 'trial_tpause': 6.0, 'trial_mi_time': 4.0,
                'trials_per_class': 72, 'eeg_channels': dv.shape[0], 'ch_labels': raw.ch_names,
                'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}

        return [dv, ev, info]

        # np.save(os.path.dirname(path) + '/A0' + str(subj) + 'E', [dv, ev, info], allow_pickle=True)


def iv2b(path, subj, ds):
    info = {'fs': 250, 'class_ids': [1, 2], 'trial_tcue': 3.0, 'trial_tpause': 7.0, 'trial_mi_time': 4.0,
            'trials_per_class': 60, 'eeg_channels': 3, 'ch_labels': ['EEG-C3', 'EEG-Cz', 'EEG-C4'],
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}

    session = '01T' if '01T' in path else '02T' if '02T' in path else '03T' if '03T' in path else '04T' if '04T' in path else '05T'

    raw = mne.io.read_raw_gdf(path).load_data()
    data = raw.get_data()[:3]  # [channels x samples]
    # data = corrigeNaN(data)
    ev = mne.events_from_annotations(raw)  # raw.find_edf_events()
    ev = np.delete(ev[0], 1, axis=1)  # elimina coluna de zeros
    truelabels = np.ravel(loadmat(os.path.dirname(path) + '/true_labels/B0' + str(subj) + session + '.mat')['classlabel'])

    if session in ['01T','02T']:
        for rm in range(5, 12): ev = np.delete(ev, np.where(ev[:, 1] == rm),
                                               axis=0)  # detele various eye movements marks
        ev = np.delete(ev, np.where(ev[:, 1] == 1), axis=0)  # delete rejected trials
        ev[:, 1] = np.where(ev[:, 1] == 2, 0, ev[:, 1])  # altera label start trial de 2 para 0
        ev[:, 1] = np.where(ev[:, 1] == 3, 1, ev[:, 1])  # altera label cue LH de 3 para 1
        ev[:, 1] = np.where(ev[:, 1] == 4, 2, ev[:, 1])  # altera label cue RH de 4 para 2

    elif session == '03T':
        for rm in range(6, 13): ev = np.delete(ev, np.where(ev[:, 1] == rm), axis=0)  # detele various eye movements marks
        ev = np.delete(ev, np.where(ev[:, 1] == 2), axis=0)  # delete rejected trials
        ev = np.delete(ev, np.where(ev[:, 1] == 1), axis=0)  # delete feedback continuous
        ev[:, 1] = np.where(ev[:, 1] == 3, 0, ev[:, 1])  # altera label start trial de 3 para 0
        ev[:, 1] = np.where(ev[:, 1] == 4, 1, ev[:, 1])  # altera label cue LH de 4 para 1
        ev[:, 1] = np.where(ev[:, 1] == 5, 2, ev[:, 1])  # altera label cue RH de 5 para 2
    else:
        for rm in range(5, 12): ev = np.delete(ev, np.where(ev[:, 1] == rm),
                                               axis=0)  # delete various eye movements marks
        ev = np.delete(ev, np.where(ev[:, 1] == 2), axis=0)  # delete rejected trials
        ev = np.delete(ev, np.where(ev[:, 1] == 1), axis=0)  # delete feedback continuous
        ev[:, 1] = np.where(ev[:, 1] == 3, 0, ev[:, 1])  # altera label start trial de 3 para 0
        ev[np.where(ev[:, 1] == 4), 1] = truelabels  # rotula momento da dica conforme truelabels

    for i in range(0, len(ev)):
        if ev[i, 1] == 0: ev[i, 1] = (ev[i + 1, 1] + 10)  # labeling start trial [11,12] according cue [1,2]

    return [data, ev, info]


def lee19(path, subj, ds):
    Fs = 1000
    S = loadmat(path)
    T = S['EEG_MI_train']
    V = S['EEG_MI_test']
    dataT = T['x'][0, 0].T
    dataV = V['x'][0, 0].T
    eventsT = np.r_[T['t'][0, 0], T['y_dec'][0, 0]].T
    eventsV = np.r_[V['t'][0, 0], V['y_dec'][0, 0]].T

    eventsV[:,0] += dataT.shape[-1]
    e = np.r_[eventsT, eventsV]
    e[:, 1] = np.where(e[:, 1] == 2, 1, 2)  # troca class_ids 1=LH, 2=RH

    d = np.c_[dataT, dataV]
    d = d[[12, 35, 13, 36, 14, 38, 39, 40, 32, 8, 9, 33, 34, 37, 28, 30]]

    info = {'fs': Fs, 'class_ids': [1, 2], 'trial_tcue': 3.0, 'trial_tpause': 7.0, 'trial_mi_time': 4.0, 'trials_per_class': 100, 'eeg_channels': 16,
            'ch_labels': list(['C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CPz', 'CP4', 'FC3', 'FC1', 'FC2', 'FC4', 'C5', 'C6', 'O1', 'O2']), 'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}

    factor = 8
    Fs = Fs // factor
    info['fs'] = Fs
    d = decimate(d, factor)
    # d = np.asarray([d[:,i] for i in range(0,d.shape[-1],factor)]).T
    # d = resample(d1, d.shape[-1]//factor, axis=-1)
    e[:, 0] = [round(e[i, 0] / factor) for i in range(e.shape[0])]

    return [d, e, info]
