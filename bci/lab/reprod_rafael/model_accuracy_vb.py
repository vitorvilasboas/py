# -*- coding: utf-8 -*-
from approach_vb import Approach

if __name__ == '__main__':
    
    # FILTER SPEC
    SAMPLING_FREQ = 250.0
    LOWER_CUTOFF = 8.
    UPPER_CUTOFF = 30.
    FILT_ORDER = 5
    # EPOCH EXTRACTION CONFIG:
    EVENT_IDS = [1, 2, 3, 4]
    T_MIN, T_MAX = 2.5, 4.5  # time before event, time after event
    CSP_N = 6
    
    # ================ TRAIN MODEL ===========================
    FOLDER = "/mnt/dados/bci_tools/dset42a/npy/"
    SUBJ = '1'
    PATH_DATA = FOLDER + 'A0' + SUBJ + 'T_data.npy'
    PATH_EVENT = FOLDER + 'A0' + SUBJ + 'T_event.npy'
    EVENT_IDS = [1, 2] # [769, 770]
    
    ap = Approach()
    
    ap.define_approach(SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX)
    
    ap.set_cal_path(PATH_DATA, PATH_EVENT)
    
    ap.set_valid_channels(range(22))
    
    crossvalscore = ap.cross_validate_model(10, 0.2)
    
    print('Crossvalidation Score {}'.format(crossvalscore))
    
    PATH_DATA = FOLDER + 'A0' + SUBJ + 'E_data.npy'
    PATH_EVENT = FOLDER + 'A0' + SUBJ + 'E_event.npy'
    ap.set_val_path(PATH_DATA, PATH_EVENT)
    autoscore = ap.train_model()
    valscore = ap.validate_model()
    print('Self Validation Score {}'.format(autoscore))
    print('Validation Score {}'.format(valscore))