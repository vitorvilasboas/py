# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas

import math
import mne
import numpy as np
from scipy.io import loadmat
import sys

sys.path.append('./reprod_rafael/')
from processor_vb import Learner, Filter

def extrairEpocas(data, eventos, classes, smin, smax):
    data = data.T[range(22)] # considera somente os 22 canais com dados EEG
    ev_tipos = eventos[:,1]
    cond = False
    for i in range(len(classes)): cond += (ev_tipos == classes[i])
    # cond é um vetor, cujo indice contém True se a posição correspondente em ev_tipos contém 1, 2, 3 ou 4
    idx = np.where(cond)[0] # contém as 288 indices que correspondem ao carimbo de uma das 4 classes 
    
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

''' 
# Windows
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

# Linux
def rotulagem(rotulos, ds, trueLabels):
    
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


def teste_load():   
    data1 = np.load('/mnt/dados/bci_tools/dset42a/npy/A01E_data.npy')
    events1 = np.load('/mnt/dados/bci_tools/dset42a/npy/A01E_event.npy')
    print(data1.shape, events1.shape)
    print(events1[:50,1])
    
def signal_processing():
    folder = '/mnt/dados/bci_tools/dset42a/npy/'
    ds = ['T','E']
    suj = 1
    
    Fs = 250.0
    classes = [1, 2]
    Tmin, Tmax = 2.5, 4.5 # Start trial= 0s ; Start dica= 2s ; End MI= 6s ;
    sample_min = int(math.floor(Tmin * Fs))
    sample_max = int(math.floor(Tmax * Fs))

    f_low, f_high = 8., 30.
    f_order = 5
    csp_nei = 6

    dataT = np.load(open(folder + 'A0' + str(suj) + ds[0] + '_data.npy', "rb"))
    dataE = np.load(open(folder + 'A0' + str(suj) + ds[1] + '_data.npy', "rb"))
    eventT = np.load(open(folder + 'A0' + str(suj) + ds[0] + '_event.npy', "rb"))
    eventE = np.load(open(folder + 'A0' + str(suj) + ds[1] + '_event.npy', "rb"))
    epocasT, rotulosT = extrairEpocas(dataT, eventT, classes, sample_min, sample_max) # [288, 22, 500], [288]
    epocasE, rotulosE = extrairEpocas(dataE, eventE, classes, sample_min, sample_max)
    
    filtro = Filter(f_low, f_high, Fs, f_order, filt_type='iir', band_type='band')
    epocasT = filtro.apply_filter(epocasT)
    epocasE = filtro.apply_filter(epocasE)
    
    learner = Learner()
    learner.design_LDA()
    learner.design_CSP(csp_nei)
    learner.assemble_learner()
    
    n_iter = 10
    test_perc = 0.2
    score = learner.cross_evaluate_set(epocasT, rotulosT, n_iter, test_perc)
    print('Crossvalidation Score {}'.format(round(score*100,2)))

    learner.learn(epocasT, rotulosT)
    
    learner.evaluate_set(epocasT, rotulosT)
    autoscore = learner.get_results() # TREINAMENTO
    print('Self Validation Score {}'.format(round(autoscore*100,2)))
    
    learner.evaluate_set(epocasE, rotulosE)
    valscore = learner.get_results() # VALIDAÇÃO
    print('Validation Score {}'.format(round(valscore*100,2)))
    
        
def gdf_to_npy():
    folder = '/mnt/dados/bci_tools/dset42a/'
    dataset = ['T','E']
    
    Fs = 250.0
    classes = [1, 2, 3, 4]
    Tmin, Tmax = 2, 6 # Start trial= 0s ; Start dica= 2s ; End MI= 6s ;
    sample_min = int(math.floor(Tmin * Fs))
    sample_max = int(math.floor(Tmax * Fs))
    
    tam_janela = sample_max - sample_min # N = número de amostras por epoca
    
    for ds in dataset:
        for suj in range(1,2):
            ### Carregando Dataset usando pacote MNE
            mne.set_log_level('WARNING','DEBUG')
            raw = mne.io.read_raw_gdf(folder + 'A0' + str(suj) + ds + '.gdf')
            raw.load_data()
            
            ### Extraindo Matriz de Dados Brutos
            dados = raw.get_data() # [p x q] [25 x 672528]
            dados = corrigeNaN(dados) # Correção de NaN nos dados brutos
            dados = dados.T # [q x p] [672528 x 25]  
            np.save(folder + 'npy/A0' + str(suj) + ds + '_data', dados) # exportando/salvando como .npy
            
            # ------------------------------------------------------------ #
            
            ### Extraindo Info Eventos
            eventsGDF = raw.find_edf_events()
            #ev_descricao= eventsGDF[1]
            ev_posicoes = eventsGDF[0][:,0]
            #ev_zeros = eventsGDF[0][:,1]
            ev_tipos = eventsGDF[0][:,2]
            #for key, values in ev_descricao.items(): print(key, ' --> ', values)
            
            ### Carregando Rótulos verdadeiros para uso em datasets de validação (E)
            trueLabels = np.ravel(loadmat(folder + 'true_labels/A0' + str(suj) + 'E.mat')['classlabel'])
            
            ### Rotulando corretamente os eventos conforme descrição da competição
            ev_tipos = rotulagem(ev_tipos,ds, trueLabels)
            
            ### Reorganizando em Matriz 603x2 timestamps e rotulos corretos
            eventos = np.asarray(np.transpose([ev_posicoes + 1, ev_tipos]))
            np.save(folder + 'npy/A0' + str(suj) + ds + '_event', eventos) # exportando/salvando como .npy
            
            # ------------------------------------------------------------ #
            
            ### Extraindo todas as 288 épocas (72 por classe)
            epocas, classlabels = extrairEpocas(dados, eventos, classes, sample_min, sample_max) # [288, 22, 1000], [288]
            epocas = nanCleaner(epocas) # Correção de NaN nas épocas
            np.save(folder + 'npy/A0' + str(suj) + ds + '_epochs', epocas) # exportando/salvando como .npy
            
            ### Extrair épocas de cada classe específica
            for label in range(1, 5):
                idx = np.where(classlabels==label)
                epocas_i = epocas[idx]
                np.save(folder + 'npy/epocas/A0' + str(suj) + ds + '_' + str(label), epocas_i)
                #print(suj,ds,label, ': ',epocas_i.shape)
               
if __name__ == '__main__':
    gdf_to_npy()  # load GDF and create NPY files
    # teste_load()
    # signal_processing() # classifier brain acitivities
    
    
    
    
    
    
    
    