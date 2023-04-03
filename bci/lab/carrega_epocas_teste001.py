# -*- coding: utf-8 -*-
import numpy as np
import mne

raw = mne.io.read_raw_gdf('/home/vboas/devto/datasets/BCICIV_2a/A01T.gdf')
raw.load_data()
dadosGDF = raw.get_data() # Extraindo Matriz de Dados Brutos (canais x amostras)
events = raw.find_edf_events() # Extraindo MetaInfo (Eventos)
eventosGDF = np.delete(events[0],1,axis=1) # | n_amostra(timestamps) | rótulo evento | 

dados_1T = np.load('/home/vboas/devto/datasets/BCICIV_2a/npy/A01T_data.npy').T


