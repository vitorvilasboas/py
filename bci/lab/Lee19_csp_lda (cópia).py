"""
-*- coding: utf-8 -*-
Created on Sat Mar 14 18:08:50 2020
@author: Vitor Vilas-Boas
"""
import math
import pickle
import numpy as np
from time import time
from scipy.stats import norm
from sklearn.svm import SVC
from scipy.io import loadmat
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from bci_utils import extractEpochs, nanCleaner, Filter, CSP
from sklearn.metrics import cohen_kappa_score 

suj = 1
fs = 250
class_ids = [1, 2]
path = '/mnt/dados/eeg_data/Lee19/'
f_low, f_high, ncomp, tmin, tmax = 8, 30, 8, 0.5, 2.5 
clf = {'model':'LDA', 'lda_solver':'svd'}
filtering = {'design':'IIR', 'iir_order':5}

smin = math.floor(tmin * fs)
smax = math.floor(tmax * fs)
buffer_len = smax - smin

filt = Filter(f_low, f_high, buffer_len, fs, filtering)
csp = CSP(n_components=ncomp)
clf_final = LDA(solver=clf['lda_solver'], shrinkage=None)

# mat = loadmat(path+'session1/sess01_subj0' + str(suj) + '_EEG_MI.mat')
# dataT = mat['EEG_MI_train']['x'][0,0].T
# dataV = mat['EEG_MI_test']['x'][0,0].T
# eventsT = np.r_[ mat['EEG_MI_train']['t'][0,0], mat['EEG_MI_train']['y_dec'][0,0] ].T
# eventsV = np.r_[ mat['EEG_MI_test']['t'][0,0], mat['EEG_MI_test']['y_dec'][0,0] ].T

data, events, info = np.load(path + 'npy/S' + str(suj) + '.npy', allow_pickle=True)

# cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
cortex = [7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 32, 33, 34, 35, 36, 37, 38, 39, 40]
data = data[cortex]
info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
info['eeg_channels'] = len(cortex)

epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
# epochs = nanCleaner(epochs)

# epochs, labels = epochs[:int(len(epochs)/2)], labels[:int(len(labels)/2)] # somente sessão 1
epochs, labels = epochs[int(len(epochs)/2):], labels[int(len(labels)/2):] # somente sessão 2

test_size = int(len(epochs) * 0.5)
train_size = int(len(epochs) - test_size)
train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
epochsT, labelsT = epochs[:train_size], labels[:train_size] 
epochsV, labelsV = epochs[train_size:], labels[train_size:]

XT = [ epochsT[np.where(labelsT == i)] for i in class_ids ] # Extrair épocas de cada classe
XV = [ epochsV[np.where(labelsV == i)] for i in class_ids ]
XT = np.concatenate([XT[0],XT[1]]) # Train data classes A + B
XV = np.concatenate([XV[0],XV[1]]) # Test data classes A + B        
yT = np.concatenate([class_ids[0] * np.ones(int(len(XT)/2)), class_ids[1] * np.ones(int(len(XT)/2))])
yV = np.concatenate([class_ids[0] * np.ones(int(len(XV)/2)), class_ids[1] * np.ones(int(len(XV)/2))])

XTF = filt.apply_filter(XT)
XVF = filt.apply_filter(XV)

csp.fit(XTF, yT)
XT_CSP = csp.transform(XTF)
XV_CSP = csp.transform(XVF) 
clf_final.fit(XT_CSP, yT)
scores = clf_final.predict(XV_CSP)
csp_filters = csp.filters_

acc = np.mean(scores == yV) # or chain.score(XVF, yV)     
kappa = cohen_kappa_score(scores, yV)

print(suj, str(round(acc*100,2))+'%', str(round(kappa,3)))