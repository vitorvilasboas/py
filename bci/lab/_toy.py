# -*- coding: utf-8 -*-
import re
import math
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from bci_utils import extractEpochs

class_ids = [1,2]
data, events, info  = np.load('/mnt/dados/eeg_data/IV2a/npy/A01T.npy', allow_pickle=True) # sessão 1
epochs, labels = extractEpochs(data, events, int(0.5*info['fs']), int(2.5*info['fs']), class_ids)
idx_a = np.where(labels == class_ids[0])[0]
k = np.random.choice(idx_a)
a = np.random.choice([0,1])

math.trunc()

U1_local = deque(maxlen=10)

U1_local.append(1)
U1_local.append(1)
U1_local.append(1)
U1_local.append(1)
U1_local.append(1)
U1_local.append(1)

#%%
# fig, ax = plt.subplots()
# ax.plot(X1[0,0,0,:10])
# plt.show()
# fig, ax = plt.subplots()
# ax.plot(XF1[0,0,0,:10])
# plt.show()
#%%
# # quebra linhas de código com \
# label_msg = "Sessão " + sname \
#    + " já existe, seus dados serão sobrescritos."
#%%
# BAUD = 115200
circBuff = deque(maxlen=500)
# tBuff = deque(maxlen=500)
# sample_counter = 0
# def print_raw(sample):
#     print('teste')
#     print(sample.channels_data)
# def GetData(sample):
#     print(sample.channels_data)
#     indata = sample.channels_data   
#     global circBuff
#     circBuff.append(indata)   
#     global sample_counter
#     tBuff.append(sample_counter)    
#     data = np.array(indata)  # transform list into numpy array
#     all_data = np.array([]).reshape(0, len(data))
#     all_data = np.vstack(all_data, data) # append to data stack    
#     time.sleep(10)
#     sample_counter += 1    
# board = pyOpenBCI.OpenBCICyton(port='/dev/ttyUSB0', baud=BAUD, daisy=False)
# board.start_stream(GetData)
# x = board.start_stream(print_raw)
#%%
# np.vstack
# data = [epochs, events]
# with open(path + suj + '.pickle', 'wb') as handle: pck.dump(data, handle)
# X, y_ = pck.load(open(file_path)) # loading pickle file type (data + labels)
# minimo = min(np.diff(pos)) # calcula o menor intervalo entre os elementos de pos
#%%
# 'vitor mendes'.replace(',','')
# teste = list(map(int, '1 2'.split(' ')))
# [ num for num in '1 2' if num.isdigit() ].replace   
# hex(ord('ã')) # descobrir sequência scape unicode de um ccaractere especial
# path = os.path.dirname(__file__) #captura o caminho do arquivo atual
# path = os.path.dirname(__file__) + '/eeg_epochs/BCI_CAMTUC/CL_'
# all_files = os.listdir('/home/vboas/cloud/devto/overmind/' + '.')
# all_pkl_files = [] 
# for fl in all_files: 
#     filename, file_extension = os.path.splitext('/home/vboas/cloud/devto/overmind/view/' + fl)
#     if file_extension == '.py': all_pkl_files.append(filename)
# last_file = max(all_pkl_files, key=os.path.getmtime).split('/')[-1]
# print(last_file)
# filename, file_extension = os.path.splitext('/home/vboas/cloud/devto/overmind/view/' + fl) # extrair nome e extensão de arquivo
# FINAL = pd.DataFrame(ACC, columns=['Suj','Classes','Acc'])
# now = datetime.now().strftime('%d-%m-%Y_%Hh%Mm')
#%%
# PATH_TO_SESSION = '/home/vboas/cloud/devto/overmind/userdata/A1'
# pattern = re.compile(r"optrial_.*?\.pkl") # mask to get filenames
# pkl_files = []
# for root, dirs, files in os.walk(PATH_TO_SESSION): # add .kv files of view/kv in kv_files vector
#     pkl_files += [root + '/' + file_ for file_ in files if pattern.match(file_)]
# try:
#     last_file = max(pkl_files, key=os.path.getmtime).split('/')[-1]  
# except:
#     print('nenhum arquivo')
# for file_ in kv_files:
#     Builder.load_file(file_) # load .kv files added