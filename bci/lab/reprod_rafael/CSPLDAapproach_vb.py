# -*- coding: utf-8 -*-
"""CSP + LDA approach.
Implements the CSP + LDA approach using a data from the V BCI competition
"""

from handler_vb import load_data_as_np, read_events, extract_epochs
from processor_vb import Filter, Learner

import math


def apply_ml(DATA_CAL_PATH, CAL_EVENTS_PATH, DATA_VAL_PATH, VAL_EVENTS_PATH,
             SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF, FILT_ORDER,
             CSP_N, EVENT_IDS, T_MIN, T_MAX):

    dp = Filter(LOWER_CUTOFF, UPPER_CUTOFF, SAMPLING_FREQ, FILT_ORDER)

    # # LOAD CALIBRATION DATA:
    # if DATA_CAL_PATH[-3:] == 'gdf':
    # 	data_cal, SAMPLING_FREQ = loadBiosig(DATA_CAL_PATH).T
    # 	data_cal = nanCleaner(data_cal)

    # else:
    data_cal = load_data_as_np(DATA_CAL_PATH).T

    events_list_cal = read_events(CAL_EVENTS_PATH)

    data_cal = dp.apply_filter(data_cal)

    # FEATURE EXTRACTION:
    SMIN = int(math.floor(T_MIN * SAMPLING_FREQ))
    SMAX = int(math.floor(T_MAX * SAMPLING_FREQ))

    epochs_cal, labels_cal = extract_epochs(
        data_cal, events_list_cal, SMIN, SMAX)

    dl = Learner()

    dl.design_LDA()
    dl.design_CSP(CSP_N)
    dl.assemble_learner()
    dl.learn(epochs_cal, labels_cal)

    data_val = load_data_as_np(DATA_VAL_PATH).T

    events_list_val = read_events(VAL_EVENTS_PATH)

    data_val = dp.apply_filter(data_val)

    # FEATURE EXTRACTION:
    epochs_val, labels_val = extract_epochs(
        data_val, events_list_val, SMIN, SMAX)

    dl.evaluate_set(epochs_val, labels_val)

    dl.print_results()

    return dl.get_results()

