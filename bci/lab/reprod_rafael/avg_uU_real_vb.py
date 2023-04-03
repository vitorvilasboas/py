# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sys
import os
import math
from random import randint

# sys.path.insert(1, os.path.join(sys.path[0], '..'))

from approach_vb import Approach


DATA_FOLDER_PATH = "/mnt/dados/bci_tools/dset42a/npy/"
EVENTS_FOLDER_PATH = "/mnt/dados/bci_tools/dset42a/npy/"

SUBJ = '8'

SAMPLING_FREQ = 250.0

# FILTER SPEC
LOWER_CUTOFF = 8.
UPPER_CUTOFF = 30.
FILT_ORDER = 7

# EPOCH EXTRACTION CONFIG:
EVENT_IDS = [1, 2, 3, 4]

T_MIN, T_MAX = 2.5, 4.5  # time before event, time after event

CSP_N = 8

# ================ TRAIN MODEL ===========================
# ========================================================
# ========================================================

DATA_PATH = DATA_FOLDER_PATH + 'A0' + SUBJ + 'T_data.npy'

# EVENTS INFO PATH
EVENTS_PATH = EVENTS_FOLDER_PATH + 'A0' + SUBJ + 'T_event.npy'
EVENT_IDS = [1, 2]

ap = Approach()
ap.define_approach(SAMPLING_FREQ, LOWER_CUTOFF, UPPER_CUTOFF,
                   FILT_ORDER, CSP_N, EVENT_IDS, T_MIN, T_MAX)

ap.set_cal_path(DATA_PATH, EVENTS_PATH)

ap.set_valid_channels(range(22))

autoscore = ap.train_model()

DATA_PATH = DATA_FOLDER_PATH + 'A0' + SUBJ + 'E_data.npy'

# EVENTS INFO PATH
EVENTS_PATH = EVENTS_FOLDER_PATH + 'A0' + SUBJ + 'E_event.npy'

ap.set_cal_path(DATA_PATH, EVENTS_PATH)
ap.set_val_path(DATA_PATH, EVENTS_PATH)
valscore = ap.validate_model()
# epochs = ap.preProcess(epochs)

crossvalscore = ap.cross_validate_model(30, 0.3)

print('-----------------------------------')
print('Crossvalidation Score {}'.format(crossvalscore))
print('Validation Score {}'.format(valscore))
print('-----------------------------------')
# print('Positive rate: {}'.format(TFNP[0] + TFNP[1]))
# print('Negative rate: {}'.format(TFNP[2] + TFNP[3]))

# ================ GENERATE FAKE DATASET =================
# ========================================================
# ========================================================

data, ev = ap.load_data(DATA_PATH, EVENTS_PATH)

epochs, labels = ap.load_epochs(data, ev)

idx_1 = np.where(labels == 1)[0]
idx_2 = np.where(labels == 2)[0]
idx_3 = np.where(labels == 3)[0]
idx_4 = np.where(labels == 4)[0]

# ================ APPEND EPOCHS =================

N_RUNS = 500
first = True
increment = 25

counter = 0
counter1 = 0
counter2 = 0

for a in range(N_RUNS):

    new_data_labels = np.array([0, 0])
    new_data = np.zeros([1, 22])

    for j in range(1):
        # add epochs from class 1 (left)
        for i in range(0, 6):
            k = randint(0, len(idx_1) - 1)
            # k = i
            new_data_labels = np.vstack(
                (new_data_labels, [1, int(new_data.shape[0])]))
            new_data = np.vstack((new_data, epochs[idx_1[k]].T))

    data, events = ap.load_data(new_data, new_data_labels, data_format='npy')

    data = data.T

    buf = np.array([data.shape[0], 500])

    u_time = np.array([])
    U_time = np.array([])
    U_max = 100
    U = 0
    i = 0
    tinit, tend = 0, 500

    while tend < data.shape[1]:

        idx = range(tinit, tend)

        buf = data[:, idx]

        p = ap.classify_epoch(buf, out_param='prob')[0]

        u = p[0] - p[1]

        if u >= 0:
            u = 1
            counter1 += 1.
        else:
            counter2 += 1.
            u = -1

        counter += 1.

        u_time = np.append(u_time, u)
        U = U + u

        # if abs(U) > U_max:
        #     U = 0
        U_time = np.append(U_time, U)

        tinit += increment
        tend += increment

    if first:
        u_avg = np.zeros(u_time.shape)
        U_avg = np.zeros(U_time.shape)
        u_h = np.zeros(u_time.shape)
        U_h = np.zeros(U_time.shape)
        first = False

    u_h = np.vstack([u_h, u_time])
    U_h = np.vstack([U_h, U_time])

    u_avg += u_time
    U_avg += U_time

u_h = u_h[1:]
U_h = U_h[1:]

u_avg = u_avg / float(N_RUNS)
U_avg = U_avg / float(N_RUNS)

# mse_u = []
# mse_U = []
# for i in range(1, u_h.shape[0]):
#     m_u = mean_squared_error(u_avg, u_h[i])
#     m_U = mean_squared_error(U_avg, U_h[i])
#     mse_u.extend([m_u])
#     mse_U.extend([m_U])

# idx_max_error_u = np.argmax(mse_u)

# u_error = u_h[idx_max_error_u]
# U_error = U_h[idx_max_error_u]


# ================ VAR ANALYSIS ==========================
# ========================================================
# ========================================================

mu = np.mean(u_avg)

u_var = np.var(u_h, axis=0)
sigma = math.sqrt(np.mean(u_var))

fake_u = np.linspace(-2, 2, 100)

# ================ PLOT RESULTS ==========================
# ========================================================
# ========================================================

print('class 1 posterior rate: {}'.format(counter1 / counter))
print('class 2 posterior rate: {}'.format(counter2 / counter))

n_samples = u_time.shape[0]

time = range(n_samples)
time = [x * increment / SAMPLING_FREQ for x in time]

U_avg_est = (valscore - (1 - valscore)) * \
    np.array(time) / (increment / SAMPLING_FREQ)

plt.plot()

plt.subplot(2, 1, 1)
plt.plot(time, U_avg, 'k',  linewidth=4.0, label='Mean')
plt.plot(time, U_avg_est, 'g', linewidth=4.0, label='Estimated')
# plt.plot(time, U_error, 'r', linewidth=.5, label='Max MSE')
plt.grid(True)
# plt.axis([0, 6, -20, 120])
# plt.axis('equal')
plt.ylabel('U')
# plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc=0)

plt.subplot(2, 1, 2)
plt.plot(time, u_avg, 'k', linewidth=3.0, label='Mean')
# plt.plot(time, u_error, 'r', linewidth=.5, label='Max MSE')
plt.grid(True)
plt.axis([0, 10, -1.2, 1.2])
# plt.axis('equal')
plt.ylabel('u')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend(loc=0)

# plt.subplot(3, 1, 3)
# plt.plot(fake_u, mlab.normpdf(fake_u, mu, sigma),
#          'k', linewidth=3.0, label='u distribution')
# # plt.plot(time, u_error, 'r', linewidth=.5, label='Max MSE')
# plt.grid(True)
# # plt.axis('equal')
# plt.ylabel('p(u)')
# plt.xlabel('u')
# plt.grid(True)
# plt.legend(loc=0)

plt.show()

