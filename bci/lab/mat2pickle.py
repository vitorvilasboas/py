import scipy.io as sio
import pickle
import numpy as np

# 118 canais
# 280 tentativas, sendo 140=1=mão direita e 140=2=pé
# 100 amostras por segundo

# Sujeitos treinamento validação (qtds total=280)
# aa tr=168 te=112
# al tr=224 te=56
# av tr=84  te=196
# aw tr=56  te=224
# ay tr=28  te=252

dset = sio.loadmat('D:/bci_tools/dset34a/matlab/aa.mat')
dados = dset['cnt'].T #transposta
classes = dset['mrk']['y'][0][0] #sequencia de classes (rótulos - NaN=dados validação)
pos = dset['mrk']['pos'][0][0] #posições amostras inicio tentativa de classe
fa = int(dset['nfo']['fs'][0][0]) #frequencia amostragem
channels = dset['nfo']['clab'][0][0] #nomes dos 118 canais

labels = sio.loadmat('D:/bci_tools/dset34a/true_labels/trues_aa.mat')
classes_true = labels['true_y']
idxv = labels['test_idx'] #indices de validação
dset['mrk']['y'][0][0] = labels['true_y']

fileName = "D:/bci_tools/dset34a/aa.pickle"  # abre o arquivo para escrever
fileObject = open(fileName, 'wb')
pickle.dump(dset, fileObject)

#print(len(dados),len(dados[0]))
#print(len(classes),len(classes[0]))
#print(len(pos),len(pos[0]))
#print(len(channels),len(channels[0]))
#print(len(idxv),len(idxv[0]))
#print(len(classes_true),len(classes_true[0]))
#print(fa)

#print(type(mat))
#print(mat.items())
#print(mat.keys())
#print(mat.values())

#x = np.array(mat['mrk']['pos'])
#print(x.shape)

