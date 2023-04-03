# -*- coding: utf-8 -*-
import numpy as np
import pickle as pck
import math
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    # aa : tr=168 te=112  
    # al : tr=224 te=56
    # av : tr=84  te=196
    # aw : tr=56  te=224
    # ay : tr=28  te=252
    
    Fs = 100
    Tmin, Tmax = 0, 5 # Start trial= 0; Start cue=0; Start MI= 0; End trial(break)= 5
    sample_min = int(math.floor(Tmin * Fs)) # amostra inicial (ex. 0)
    sample_max = int(math.floor(Tmax * Fs)) # amostra final (ex. 2500)
    
    folder = '/mnt/dados/datasets/BCICIII_4a/'
    
    subjects = ['aa','al','av','aw','ay'] 
    
    for suj in subjects:
        mat = loadmat(folder + suj + '.mat')
        cnt = 0.1 * mat['cnt'] # convert to uV
        pos = mat['mrk'][0][0][0][0]
        y = mat['mrk'][0][0][1][0]
        
        true_mat = loadmat(folder + 'truelabels/trues_' + suj + '.mat')
        true_y = np.ravel(true_mat['true_y']) # RH=1 Foot=2
        true_y = np.where(true_y==2,3,true_y) # RH=1 Foot=3
        # true_test_idx = np.ravel(true_mat['test_idx'])
        
        epochs = [ cnt[p+sample_min:p+sample_max] for p in pos ]
        
        epochs = np.asarray(epochs).transpose(0,2,1)
        
        data = [epochs, true_y]
        with open(folder + 'pickle/' + suj + '.pickle', 'wb') as handle:
             pck.dump(data, handle)
        
        for label in [1,3]: # 1=RH e 
            idx1 = np.where(true_y==label)
            epocas_i = epochs[idx1]
            np.save(folder + 'npy/' + suj + '_' + str(label), epocas_i) # 1 arquivo .npy contm as 140 pocas de 1 classe)
        
        # print(min(np.diff(pos)))
        
        # np.save('/mnt/dados/datasets/BCICIII_4a/'+subject+'_data', epochs)
        # np.save('/mnt/dados/datasets/BCICIII_4a/'+subject+'_labels', true_y)
        


    # file_path = folder + 'pickle/' + subjects[0] + ".pickle"
    # X, y_ = pck.load(open(file_path)) # loading pickle file type (2 classes together)
    
    classes = [1,3]
    X1 = np.load(folder + 'npy/' + subjects[0] + '_' + classes[0] + '.npy') # loading npy filetype (1 classe only)
    X3 = np.load(folder + 'npy/' + subjects[0] + '_' + classes[1] + '.npy') 






  
    
    

