# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import math
import mne
import numpy as np
from scipy.io import loadmat

def extrairEpocas(data, eventos, classes, smin, smax):
    data = data[range(22)] # considera somente os 22 canais com dados EEG
    ev_tipos = eventos[:,1]
    cond = False
    for classe in classes: cond += (ev_tipos == classe)
    # cond é um vetor, cujo indice contém True se a posição correspondente em ev_tipos contém 1, 2, 3 ou 4
    idx = np.where(cond)[0] # contém os 288 indices que correspondem ao rótulo de uma das 4 classes 
    
    t0_stamps = eventos[idx, 0] #contém os sample_stamp(posições) relacionadas ao inicio das 288 tentativas de epocas das classes
    #for pos in t0_stamps: print(pos)
    
    sBegin = t0_stamps + smin # vetor que marca as amostras que iniciam cada época
    sEnd = t0_stamps + smax # vetor que contém as amostras que finalizam cada epoca
    
    n_epochs = len(sBegin)
    n_channels = data.shape[0] # dimensão de data deve ser [672528 x 25]
    n_samples = smax - smin
    epochs = np.zeros([n_epochs, n_channels, n_samples])
    
    classlabels = ev_tipos[idx] # vetor que contém os indices das 288 épocas das 4 classes

    bad_epoch_list = []
    for i in range(n_epochs):
        epoch = data[:, sBegin[i]:sEnd[i]]
        
        if epoch.shape[1] == n_samples: # Check if epoch is complete
            epochs[i, :, :] = epoch 
        else:
            print('Incomplete epoch detected...')
            bad_epoch_list.append(i)

    classlabels = np.delete(classlabels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    
    return epochs, classlabels # retorna as 288 épocas e os indices de cada uma (labels)

def rotulagem(rotulos, ds, trueLabels): # normaliza rotulos de eventos conforme descrição oficial do dataset
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
                if rotulos[i+1] == 1023: rotulos[i] = rotulos[i+2] - rotulos[i] # (769,770,771 ou 772) - 768 = 1,2,3 ou 4
                else: rotulos[i] = rotulos[i+1] - rotulos[i] # a partir da proxima tarefa (769,770,771 ou 772) - 768 = 1,2,3 ou 4
        
    else: # if Evaluate dataset (A0sE.gdf) 
        rotulos = np.where(rotulos==5, 277, rotulos) # Idling EEG (eyes closed) 
        rotulos = np.where(rotulos==6, 276, rotulos) # Idling EEG (eyes open) 
        rotulos = np.where(rotulos==7, 32766, rotulos) # Start of a new run/segment (after a break)
        
        # muda padrão dos rotulos desconhecidos de 4 para 783 conforme descrição oficial do dataset
        idx4 = np.where(rotulos==4)
        rotulos[idx4] = trueLabels + 768
        
        # rotula inicios de trials no dset de validação conforme rótulos verdadeiros fornecidos (truelabels)
        idx768 = np.where(rotulos==768)
        rotulos[idx768] = trueLabels
        
    return rotulos

def corrigeNaN(dados):
    for canal in range(dados.shape[0] - 1):
        this_chan = dados[canal]
        dados[canal] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(dados[canal])
        mediaCanal = np.nanmean(dados[canal])
        dados[canal, mask] = mediaCanal
    return dados

def nanCleaner(epoca):
    """Removes NaN from data by interpolation
    data_in : input data - np matrix channels x samples
    data_out : clean dataset with no NaN samples"""
    for i in range(epoca.shape[0]):
        bad_idx = np.isnan(epoca[i, :])
        epoca[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoca[i, ~bad_idx])
    return epoca

                       
if __name__ == '__main__':
    
    # load GDF and create NPY files
    folder = '/mnt/dados/datasets/BCICIV_2a/'
    dataset = ['T','E']
    classes = [1, 2, 3, 4]
    sujeitos = range(1,10)
    
    Fs = 250.0
    
    Tmin, Tmax = 0, 7.5 # Start trial= 0; Start cue= 2; Start MI= 3.25; End MI= 6; End trial= 7.5-8.5
    sample_min = int(math.floor(Tmin * Fs)) # amostra inicial (ex. 0)
    sample_max = int(math.floor(Tmax * Fs)) # amostra final (ex. 1875)

    for ds in dataset:
        for suj in sujeitos:
            ### Carregando Dataset usando pacote MNE
            mne.set_log_level('WARNING','DEBUG')
            raw = mne.io.read_raw_gdf(folder + 'A0' + str(suj) + ds + '.gdf')
            raw.load_data()
            
            ### Extraindo Matriz de Dados Brutos
            dados = raw.get_data() # [p x q] [25 x 672528]
            dados = corrigeNaN(dados) # Correção de NaN nos dados brutos
            
            ### Extraindo Info Eventos
            eventsGDF = raw.find_edf_events()
            ev_posicoes = eventsGDF[0][:,0]
            ev_tipos = eventsGDF[0][:,2]
            
            ### Carregando Rótulos verdadeiros para uso em datasets de validação (E)
            trueLabels = np.ravel(loadmat(folder + 'true_labels/A0' + str(suj) + 'E.mat')['classlabel'])
            
            ### Rotulando corretamente os eventos conforme descrição da competição
            ev_tipos = rotulagem(ev_tipos,ds, trueLabels)
            
            ### Reorganizando em Matriz 603x2 timestamps e rotulos corretos
            eventos = np.asarray(np.transpose([ev_posicoes, ev_tipos]))
            
            ### Extraindo todas as 288 épocas (72 por classe)
            epocas, classlabels = extrairEpocas(dados, eventos, classes, sample_min, sample_max) # [288, 22, 1000], [288]
            #epocas = nanCleaner(epocas) # Correção de NaN nas épocas
            
# =============================================================================
#             data = dados[range(22)] # considera somente os 22 canais com dados EEG
#             ev_tipos = eventos[:,1]
#             cond = False
#             for classe in classes: cond += (ev_tipos == classe)
#             # cond é um vetor, cujo indice contém True se a posição correspondente em ev_tipos contém 1, 2, 3 ou 4
#             idx = np.where(cond)[0] # contém os 288 indices que correspondem ao rótulo de uma das 4 classes 
#             t0_stamps = eventos[idx, 0] #contém os sample_stamp(posições) relacionadas ao inicio das 288 tentativas de epocas das classes
#             sBegin = t0_stamps + sample_min # vetor que marca as amostras que iniciam cada época
#             sEnd = t0_stamps + sample_max # vetor que contém as amostras que finalizam cada epoca
#             n_epochs = len(sBegin)
#             n_channels = data.shape[0] # dimensão de data deve ser [672528 x 25]
#             n_samples = sample_max - sample_min
#             epocas = np.zeros([n_epochs, n_channels, n_samples])
#             classlabels = ev_tipos[idx] # vetor que contém os indices das 288 épocas das 4 classes
#             bad_epoch_list = []
#             for i in range(n_epochs):
#                 epoch = data[:, sBegin[i]:sEnd[i]]
#                 if epoch.shape[1] == n_samples: # Check if epoch is complete
#                     epocas[i, :, :] = epoch 
#                 else:
#                     print('Incomplete epoch detected...')
#                     bad_epoch_list.append(i)
#             classlabels = np.delete(classlabels, bad_epoch_list)
#             epocas = np.delete(epocas, bad_epoch_list, axis=0)
# =============================================================================
            
            #epocas = nanCleaner(epocas) # Correção de NaN nas épocas
            
            ### Extrair épocas específicas de cada classe para cada sujeito e seçao (T ou E)
            for label in range(1, 5):
                idx1 = np.where(classlabels==label)
                epocas_i = epocas[idx1]
                np.save(folder + 'npy/A0' + str(suj) + ds + '_' + str(label), epocas_i) # 1 arquivo .npy contm as 72 pocas de 1 classe)
            
            ### Salva em .npy : dados_brutos, eventos, epocas e seus respectivos rótulos
            #np.save(folder + 'npy/A0' + str(suj) + ds + '_data', dados)
            #np.save(folder + 'npy/A0' + str(suj) + ds + '_event', eventos)
            #np.save(folder + 'npy/A0' + str(suj) + ds + '_epochs', epocas)
            #np.save(folder + 'npy/A0' + str(suj) + ds + '_epochs_labels', classlabels)
    
    
    suj01 = np.load('/home/vboas/devto/datasets/BCICIV_2a/npy/A01T_1.npy') # teste de carregamento
    
    