# -*- coding: utf-8 -*-
# arquivos necessários: processor_vb.py e algorithms_vb.py
import mne
import math
import numpy as np
import scipy.signal as sp
from scipy.io import loadmat
import sys

sys.path.append('./reprod_rafael/')
from processor_vb import Learner, Filter

class LoadDataset:
    
    def nanCleaner(self, d):
        """Removes NaN from data by interpolation
        data_in : input data - np matrix channels x samples
        data_out : clean dataset with no NaN samples"""
        for i in range(d.shape[0]):
            bad_idx = np.isnan(d[i, :])
            d[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], d[i, ~bad_idx])
        return d
    
    def extract_epochs(self, data, e, smin, smax, ev_id):
        events_list = e[:, 1]
        cond = False
    
        for i in range(len(ev_id)):
            cond += (events_list == ev_id[i])
    
        idx = np.where(cond)[0] # contém as 288 posições das 4 classes 
        s = e[idx, 0] # contém os sample_stamp relacionadas ao inicio das 288 epocas das classes
    
        sBegin = s + smin # vetores que marcam a amostra que iniciam e finalizam cada época
        sEnd = s + smax
        
        data = data.T[range(22)] #data[range(22)].T se transpõs antes, no load, tem que transpor aki
        #print(data.shape)
        n_epochs = len(sBegin)
        n_channels = data.shape[0]
        n_samples = smax - smin
        
        epochs = np.zeros([n_epochs, n_channels, n_samples])
    
        labels = events_list[idx] # vetor que contém os indices das 288 épocas das 4 classes
    
        bad_epoch_list = []
        for i in range(n_epochs):
            epoch = data[:, sBegin[i]:sEnd[i]]
            # Check if epoch is complete
            if epoch.shape[1] == n_samples:
                epochs[i, :, :] = epoch
            else:
                print('Incomplete epoch detected...')
                bad_epoch_list.append(i)
    
        labels = np.delete(labels, bad_epoch_list)
        epochs = np.delete(epochs, bad_epoch_list, axis=0)
        
        return epochs, labels # retorna as 288 épocas e os indices de cada uma (labels)

    def load_NPY(self, path_data, path_events): # load .npy file
        data1T = np.load(open(path_data, "rb"))
        events1T = np.load(path_events)
        
        return data1T, events1T
        #pos = events[:,0] 
        #tipo = events[:,1] # [:,]todas as linhas, coluna [,Y]
        #print(tipo, cont)
        
        #t_events = np.insert(events, 1, values=0, axis=1) # insert dummy column to fit mne event list format
        #t_events = t_events.astype(int)  # convert to integer
        
        #cont = 0
        #for i in range(len(tipo)):
        #    if (tipo[i] == 769) or (tipo[i] == 770) or (tipo[i] == 771) or (tipo[i] == 772): cont += 1
            
    def load_MAT_trueLabels(self, sujeito):
        mat = loadmat('/mnt/dados/bci_tools/dset42a/true_labels/A0' + str(sujeito) + 'E.mat')
        y = np.ravel(loadmat('/mnt/dados/bci_tools/dset42a/true_labels/A0' + str(sujeito) + 'E.mat')['classlabel'])
        return y, mat
    
    
    def load_GDF(self, path): # load .gdf file
        mne.set_log_level("WARNING")
        raw_edf = mne.io.read_raw_gdf(path) # , stim_channel='auto', preload=True)
        return raw_edf
  
      
    def correctNaN_GDF(self, raw_edf):
        # correct nan values
        raw_edf.load_data()
        data = raw_edf.get_data()
        # do not correct stimulus channel
        # assert raw_edf.ch_names[-1] == 'STI 014'
        for i_chan in range(data.shape[0] - 1):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[i_chan]
            data[i_chan] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            data[i_chan, mask] = chan_mean
        gdf_events = raw_edf.find_edf_events()
        raw_edf = mne.io.RawArray(data, raw_edf.info, verbose='WARNING')
        # remember gdf events
        raw_edf.info['gdf_events'] = gdf_events
        return raw_edf
    
    
    '''    # windows
    def labeling_GDF(self, event, ds, suj):
        tipo = event[:,2]
        # concertando rótulos conforme originais dos datasets de treinamento e avaliação
        tipo = np.where(tipo==1, 32766, tipo) # Start of a new run/segment (after a break)
        tipo = np.where(tipo==2, 276, tipo) # Idling EEG (eyes open)
        tipo = np.where(tipo==3, 277, tipo) # Idling EEG (eyes closed)
        tipo = np.where(tipo==4, 1072, tipo) # Eye movements
        tipo = np.where(tipo==5, 768, tipo) # Start trial t=0
         
        if ds=='T': # if Training dataset (A0sT.gdf)
            tipo = np.where(tipo==6, 772, tipo) # Tongue (classe 4)
            tipo = np.where(tipo==7, 771, tipo) # Foot (classe 3)
            tipo = np.where(tipo==8, 770, tipo) # RH (classe 2)
            tipo = np.where(tipo==9, 769, tipo) # LH (classe 1)
            tipo = np.where(tipo==10, 1023, tipo) # Rejected trial
        else: # if Evaluate dataset (A0sE.gdf) 
            tipo = np.where(tipo==7, 1023, tipo) # Rejected trial
            y, mat = self.load_MAT_trueLabels(suj) # carrega true_labels
            j = 0
            y = np.where(y==1, 769, y) # concerta true_labels conforme ids 769-772
            y = np.where(y==2, 770, y)
            y = np.where(y==3, 771, y)
            y = np.where(y==4, 772, y)
            for i in range(len(tipo)): # rotula event_type a partir do vetor de truelabels
                if tipo[i] == 6:
                    tipo[i] = y[j]
                    j += 1
        
        event[:,2] = tipo
        return 
    '''
    
    def labeling_GDF(rotulos, ds, trueLabels):
    
        rotulos = np.where(rotulos==1, 1023, rotulos) # Rejected trial
        rotulos = np.where(rotulos==2, 768, rotulos) # Start trial t=0
        rotulos = np.where(rotulos==3, 1072, rotulos) # Eye movements / Unknown
        
        if ds=='T': # if Training dataset (A0sT.gdf)
            rotulos = np.where(rotulos==8, 277, rotulos) # Idling EEG (eyes closed) 
            rotulos = np.where(rotulos==9, 276, rotulos) # Idling EEG (eyes open) 
            rotulos = np.where(rotulos==10, 32766, rotulos) # Start of a new run/segment (after a break) 
            rotulos = np.where(rotulos==4, 769, rotulos) # LH (classe 1) 
            rotulos = np.where(rotulos==5, 770, rotulos) # RH (classe 2) 
            rotulos = np.where(rotulos==6, 771, rotulos) # Foot (classe 3)
            rotulos = np.where(rotulos==7, 772, rotulos) # Tongue (classe 4)
            
            for i in range(0, len(rotulos)): 
                if rotulos[i]==768: # rotula [1 a 4] o inicio da trial...
                    if rotulos[i+1] == 1023: rotulos[i] = rotulos[i+2] - rotulos[i]
                    else: rotulos[i] = rotulos[i+1] - rotulos[i] # a partir da proxima tarefa [ 1 para 769, 2 para 770... ]
            
        else: # if Evaluate dataset (A0sE.gdf) 
            rotulos = np.where(rotulos==5, 277, rotulos) # Idling EEG (eyes closed) 
            rotulos = np.where(rotulos==6, 276, rotulos) # Idling EEG (eyes open) 
            rotulos = np.where(rotulos==7, 32766, rotulos) # Start of a new run/segment (after a break)
            
            idx4 = np.where(rotulos==4)
            rotulos[idx4] = trueLabels + 768
            
            idx768 = np.where(rotulos==768)
            rotulos[idx768] = trueLabels
            
        return rotulos
    
    
    def save_gdf_to_npy(self):
        folder = '/mnt/dados/bci_tools/dset42a/'
        ds = ['T','E']
        teste = []
        for suj in range(1, 10): # (1, 10)
            for i in range(2): # (2)
                pathIN = folder + 'A0' + str(suj) + ds[i] + '.gdf'
                raw_loaded = self.load_GDF(pathIN)
                raw = self.correctNaN_GDF(raw_loaded)
                teste.append(pathIN)
                event_desc = raw.info['gdf_events'][1]
                #event_pos = raw.info['gdf_events'][0][:,0]
                #event_type = raw.info['gdf_events'][0][:,2]
                event = raw.info['gdf_events'][0][:,0:3]
                event = self.labeling_GDF(event, ds[i], suj)
                #print(event_type, event_desc)
                #eventos = np.array(list(zip(event))) # 603 x 3
                
                dados = raw.get_data() # raw.get_data().T 672528 x 25
                
                pathOUT_event = folder + 'npy2/' + 'A0' + str(suj) + ds[i] + '_event'
                pathOUT_data = folder + 'npy2/' + 'A0' + str(suj) + ds[i] + '_data'
                np.save(pathOUT_event, event)
                np.save(pathOUT_data, dados)
        #print(teste)
    
    def __init__(self):
        #self.save_gdf_to_npy()
        
        folder = '/mnt/dados/bci_tools/dset42a/npy/'
        
        sujeito = 1 # np.arange(1,10)
        ds = ['T', 'E']
        step = ['_data', '_event']
        
        path_data = folder + 'A0' + str(sujeito) + ds[0] + step[0] + '.npy'
        path_events = folder + 'A0' + str(sujeito) + ds[0] + step[1] + '.npy'
        
        data1T, events1T = self.load_NPY(path_data, path_events)
        
        Fs = 250.0
        classes = [1, 2] # 1=769, 2=770, 3=771, 4=772
        Tmin, Tmax = 2.5, 4.5
        smin = int(math.floor(Tmin * Fs))
        smax = int(math.floor(Tmax * Fs))
        
        epochs, labels = self.extract_epochs(data1T, events1T, smin, smax, classes) # [288, 22, 500], [288]
        
        data = self.nanCleaner(epochs)
        
        f_low, f_high = 8., 30.
        f_order = 5
        self.filter = Filter(f_low, f_high, Fs, f_order, filt_type='iir', band_type='band')
        
        epochs = self.filter.apply_filter(data)
        
        csp_nei = 6
        self.learner = Learner()
        self.learner.design_LDA()
        self.learner.design_CSP(csp_nei)
        self.learner.assemble_learner()
        
        n_iter = 10
        test_perc = 0.2
        
        score = self.learner.cross_evaluate_set(epochs, labels, n_iter, test_perc)
        print('Crossvalidation Score {}'.format(score))
        
        #### TREINAMENTO ####
        self.learner.learn(epochs, labels)
        self.learner.evaluate_set(epochs, labels)
        autoscore = self.learner.get_results()
        print('Self Validation Score {}'.format(autoscore))
        
        #### VALIDAÇÃO ####
        path_data = folder + 'A0' + str(sujeito) + ds[1] + step[0] + '.npy'
        path_events = folder + 'A0' + str(sujeito) + ds[1] + step[1] + '.npy'
        
        data1E, events1E = self.load_NPY(path_data, path_events)
        
        epochs, labels = self.extract_epochs(data1E, events1E, smin, smax, classes)
        
        data = self.nanCleaner(epochs)
        
        epochs = self.filter.apply_filter(data)
        
        self.learner.evaluate_set(epochs, labels)
        valscore = self.learner.get_results()
        
        print('Validation Score {}'.format(valscore))
        
if __name__ == '__main__':
    LoadDataset()
