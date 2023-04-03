# -*- coding: utf-8 -*-
import mne
import numpy as np
from scipy.io import loadmat

class LoadDataset:
    
    def load_NPY(self): # load .npy file
        #data = np.load('D:/PPCA/tools/dset_iv2a/npy/A01_rafael/data_cal.npy') # 672528 x 25
        #events = np.load('D:/PPCA/tools/dset_iv2a/npy/A01_rafael/events_cal.npy') # 603 x 2
        data = np.load('D:/PPCA/tools/dset_iv2a/npy/A01E_data.npy')
        events = np.load('D:/PPCA/tools/dset_iv2a/npy/A01E_event.npy')
        pos = events[:,0] 
        tipo = events[:,1] # [:,]todas as linhas, coluna [,Y]
        cont = 0
        for i in range(len(tipo)):
            if (tipo[i] == 769) or (tipo[i] == 770) or (tipo[i] == 771) or (tipo[i] == 772): cont += 1
        print(tipo, cont)
        
    def load_MAT_trueLabels(self, sujeito):
        mat = loadmat('D:/PPCA/tools/dset_iv2a/true_labels/A0' + str(sujeito) + 'E.mat')
        y = np.ravel(loadmat('D:/PPCA/tools/dset_iv2a/true_labels/A0' + str(sujeito) + 'E.mat')['classlabel'])
        return y, mat
    
    def load_GDF(self, path): # load .gdf file
        mne.set_log_level("WARNING", )
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
        
    def labeling_GDF(self, tipo, ds, suj):
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

        return tipo
    
    def __init__(self):
        #self.load_NPY()
    
        folder = 'D:/PPCA/tools/dset_iv2a/'
        ds = ['T','E']
        for suj in range(1, 10): # (1, 10)
            for i in range(2): # (2)
                pathIN = folder + 'A0' + str(suj) + ds[i] + '.gdf'
                print(pathIN)
                raw_loaded = self.load_GDF(pathIN)
                raw = self.correctNaN_GDF(raw_loaded)
                
                event_desc = raw.info['gdf_events'][1]
                event_pos = raw.info['gdf_events'][0][:,0]
                event_type = raw.info['gdf_events'][0][:,2]
                #print(event_type, event_desc)
                event_typeOK = self.labeling_GDF(event_type, ds[i], suj)
                eventos = np.array(list(zip(event_pos, event_typeOK))) # 603 x 2
                dados = raw.get_data().T # 672528 x 25
                
                pathOUT_event = folder + 'npy/' + 'A0' + str(suj) + ds[i] + '_event'
                pathOUT_data = folder + 'npy/' + 'A0' + str(suj) + ds[i] + '_data'
                np.save(pathOUT_event, eventos)
                np.save(pathOUT_data, dados)
        
if __name__ == '__main__':
    LoadDataset()
    

    #a = np.arange(10)
    #b = np.where(a==7, 0, a) # condição, valor pra True, valor pra False
    
    #print(raw.info['gdf_events'])
    #events = np.array(list(zip(raw.info['gdf_events'][0],raw.info['gdf_events'][1])))
    #print(raw.info["ch_names"])
    #raw.rename_channels(lambda s: s.strip("EEG-"))
    #print(raw.info["ch_names"][:10])
    #print(mne.channels.get_builtin_montages())
    #montage = mne.channels.read_montage("standard_1020")
    #montage.plot()
    #raw.set_montage(montage)
    
    #dataGDF = np.fromfile(open(D:/PPCA/tools/dset_iv2a/A01T.gdf, 'rb'), np.dtype('i'))
