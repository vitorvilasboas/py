# @by Vitor Mendes Vilas Boas - 14/Feb/2019


import numpy as np

folder = "D:/OneDrive/BCI_tools/bcic4.2a/epocas"

def load_data(SUBJECT, classes):
    X = [[], []]
    set = ['T_', 'E_']
    """
    for i in range(2):
        for j in range(2):
            path = folder + '/A0' + str(SUBJECT) + set[j] + str(classes[i]) + '.fdt'
            fid = open(path, 'rb')
            data = np.fromfile(fid, np.dtype('f'))
            data = data.reshape((72, 1000, 22))
            X[j].append(np.transpose(data, (0,2,1)))
    """
    path = folder + '/A01E_1.fdt'
    fid = open(path, 'rb')
    data = np.fromfile(fid, np.dtype('f'))
    data = data.reshape((72, 1000, 22))
    print(data[71][999][21])
    X = np.transpose(data,(0, 2, 1))
    print(X[71][21][999])


    for i in range(72):
        print('\n[ ',i,']\n')
        for j in range(1000):
            print('\n[ ',j,']\n')
            for k in range(22):
                print(data[i][j][k], end="  ")


    return X

X = load_data(1, [3, 4])

"""
import os.path as op
import numpy as np
import mne

mne.set_log_level("WARNING")
raw = mne.io.read_raw_edf("D:/OneDrive/BCI_tools/bcic4.2a/A08T.gdf", preload=True)
# event = mne.find_events(raw)
# print(raw.info)

#raw_edf = mne.io.read_raw_edf(raw, stim_channel='auto')
events = mne.find_events(raw, stim_channel='auto')

import os
mne.io.Raw.plot(raw)
# raw.plot()
os.system("PAUSE")

import matplotlib.pyplot as plt
x = [1,2,3,4,5,6,7,8]
y = [0.1,0.2,0.5,0.6,0.8,0.8,0.9,1.3]
plt.plot(x,y)
plt.show()
"""

# raw.rename_channels(lambda s: s.strip("EEG-"))
# print(raw.info["ch_names"][:10])
# print(raw.info)
# events = mne.find_events(raw, stim_channel='4')
# print('Number of events:', len(events))
# raw = mne.io.RawArray

"""
#data, sfreq = loadBiosig(data_eval_path)
raw = mne.io.RawArray(data, info)
csv_path = "/PATH/TO/CSVFILE/events.csv"
raw = addEvents(raw, eval_events_path)
event_id = dict(LH=769, RH=770)
tmin, tmax = 1, 3 # epoch starts 1 sec after event and ends 3 sec after
epochs_train, labels_train = extractEpochs(raw, event_id, tmin, tmax)
"""

"""
import numpy as np
folder = "D:/OneDrive/BCI_tools/bcic4.2a"
path = folder + '/A01T.gdf'
fid = open(path, 'rb')
data = np.fromfile(fid, np.dtype('i'))
print(data)
"""

"""
# Carregando os dados de treinamento de um só classe - Épocas já extraidas
import numpy as np
folder = "D:/OneDrive/BCI_tools/bcic4.2a/epocas"
path = folder + '/A01T_3.fdt'
fid = open(path, 'rb')
data = np.fromfile(fid, np.dtype('f'))
data = data.reshape((72, 500, 25))
X = np.transpose(data, (0,2,1))
print(X)
"""

"""
import mne
#mne.set_log_level("WARNING")
#raw = mne.io.read_raw_edf("C:/Users/vitor/Downloads/S001R04.edf", preload=True)
raw = mne.io.read_raw_edf("D:/OneDrive/BCI_tools/bcic4.2a/A08T.gdf", preload=True)

#print(raw2.info)
#print(mne.__version__)
#print(raw.info["ch_names"])
#raw.rename_channels(lambda s: s.strip("EEG-"))
#print(raw.info["ch_names"][:10])

#print(mne.channels.get_builtin_montages())
#montage = mne.channels.read_montage("standard_1020")
#montage.plot()
#raw.set_montage(montage)
"""