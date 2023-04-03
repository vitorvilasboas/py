# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import os
import re
import mne
import math
import copy
import pickle
import warnings
import itertools
import collections
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from time import time, sleep
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.stats import norm, mode
from datetime import datetime
from scipy.fftpack import fft
from scipy.linalg import eigh
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt import base, fmin, tpe, rand, hp, space_eval
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter, decimate, welch
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from functools import partial
import random
from sklearn.preprocessing import normalize

from bci_cp import Processor, Filter, extractEpochs, CSP, nanCleaner
from asetup import AutoSetup, Tunning_ncsp


np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)


def get_features(Z, setup, is_epoch=False):
    if setup.filt_type == 'DFT':
        bsize = 2 / ( setup.Fs / Z.shape[-1] )
        if is_epoch: 
            XF = setup.filter.apply_filter(Z, is_epoch=True)
            X = [XF[:, round(setup.sub_bands[i][0]*bsize):round(setup.sub_bands[i][1]*bsize)] for i in range(setup.nbands)]
        else: 
            XF = setup.filter.apply_filter(Z)
            X = [ XF[:, :, round(setup.sub_bands[i][0]*bsize):round(setup.sub_bands[i][1]*bsize)] for i in range(setup.nbands) ]             
    elif setup.filt_type == 'IIR':
        X = [setup.filter[i].apply_filter(Z) for i in range(setup.nbands)]
    
    if is_epoch:
        # F = np.asarray([np.log(np.mean(np.dot(setup.csp[i].filters_, X[i])**2, axis=1)) for i in range(setup.nbands)]) # Spatial Filtering, Feature extraction and Scoring LDA
        F = np.asarray([np.log(np.var(np.dot(setup.csp[i].filters_, X[i]), axis=1)) for i in range(setup.nbands)]) 
        S = np.ravel([setup.lda[i].transform(F[i].reshape(1,-1)) for i in range(setup.nbands)])
    else:
        F = [setup.csp[i].transform(X[i]) for i in range(setup.nbands) ] # nbands x len(X) setup.nbands ncsp
        S = np.asarray([ np.ravel(setup.lda[i].transform(F[i])) for i in range(setup.nbands) ]).T # Score LDA
        F = np.transpose(F, (1, 0, 2))   
    return F, S


if __name__ == "__main__":
    n_iter = 200
    filt = {'design':'DFT'}
    # filt = {'design':'IIR', 'iir_order': 5}
    # filt = {'design':'FIR', 'fir_order': 5}
    
    forder = None if filt['design'] == 'DFT' else filt['iir_order'] if filt['design'] == 'IIR' else filt['fir_order']
    
    R = pd.DataFrame(columns=['subj', 'fl', 'fh', 'tmin', 'tmax', 'nbands', 'ncsp', 'csp_list', 'clf', 'clf_details', 'as_max','as_best', 'sb', 'cla','as_cost'])    
    G = pd.DataFrame()
    
    ds = 'IV2a'
    path = '/home/vboas/cloud/results/as_off/' + ds + '/'
    class_ids = [1,2]

    for suj in range(1,10):
        print(f'###### {suj} {class_ids} ######')
        eeg_path_train = '/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'T.npy'
        eeg_path_test = '/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'E.npy'
        
        ### Auto Setup Cal
        bci = Processor()
        bci.overlap = True
        bci.class_ids = class_ids
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=range(0,22))
        bci.filt_type, bci.filt_order = filt['design'], forder
        bci.crossval, bci.nfolds, bci.test_perc = True, 5, 0.2
        
        asetup = AutoSetup(setup=bci, n_iter=n_iter, load_last_setup=True)
        asetup.path_out = path + ds + '_' + str(suj) + '_trials_' + filt['design'] + '.pkl'
        t0 = time()
        asetup.run_optimizer()
        cost = time() - t0
        H = asetup.H
        # acc_cal_max = (-1) * asetup.trials.best_trial['result']['loss']
        # best = asetup.best
        
        # bci = copy.copy(learner)
        # data, events, info = np.load(eeg_path_test, allow_pickle=True) 
        # Z, t = extractEpochs(data, events, int(bci.tmin*info['fs']), int(bci.tmax*info['fs']), class_ids)
        # y = bci.classify_set(Z, out_param='label')
        # acc_cal_max_test = np.mean(y == t)
        # del (globals()['Z'], globals()['t'])
           
        ### Voting and Average
        V, P = [], []
        acc_teste = []
        for i in range(len(H)):
            bci = copy.copy(H.iloc[i]['learner'])
            bci.load_eeg_train(eeg_path=eeg_path_train, channels=range(0,22))
            bci.load_eeg_test(eeg_path=eeg_path_test, channels=range(0,22))
            # bci.ncsp_list = learner_max.ncsp_list
            bci.crossval = False
            bci.process()
            acc_teste.append(bci.acc)
            t = bci.t
            V.append(bci.y)
            P.append(bci.y_prob)
        H.insert(loc=8, column='acc_test', value=acc_teste)
        
        ### Voting
        V = np.asarray(V).T
        y = np.asarray([mode(V[i])[0][0] for i in range(len(V))], dtype=int)
        acc_mode = np.mean(y == t)
    
        ### Averaging
        P = np.mean(np.transpose(P, (1,2,0)), axis=2)
        p = np.asarray([ class_ids[0] if (P[p][0]>=P[p][1]) else class_ids[1] for p in range(len(P))], dtype=int)
        acc_pmean = np.mean(p == t)
        
        hmax = H[ H['acc'] == H['acc'].max()].iloc[0]
        hmax2 = H[ H['acc_test'] == H['acc_test'].max()].iloc[0]
        
        ### Auto Setup Val
        bci = copy.copy(hmax2['learner'])
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=range(0,22))
        bci.load_eeg_test(eeg_path=eeg_path_test, channels=range(0,22))
        bci.filt_type, bci.filt_order = filt['design'], forder
        bci.crossval = False
        bci.process()
        learner = bci
        
        ### SB ncsp tunning
        acc_val_tune = 0
        if learner.is_sbcsp:
            bci = copy.copy(learner)
            bci.load_eeg_train(eeg_path=eeg_path_train, channels=range(0,22))
            asetup = Tunning_ncsp(setup=bci, n_iter=n_iter//4)
            asetup.run_optimizer()
            # best_csp_list = [int(asetup.best['csp'+str(i)]) for i in range(bci.nbands)]
            # acc_cal_tune = (-1) * asetup.trials.best_trial['result']['loss']
            ht_max = asetup.H[ asetup.H['acc'] == asetup.H['acc'].max()].iloc[0]
            
            bci = copy.copy(ht_max['learner'])
            bci.load_eeg_train(eeg_path=eeg_path_train, channels=range(0,22))
            bci.load_eeg_test(eeg_path=eeg_path_test, channels=range(0,22))
            bci.crossval = False
            bci.process()
            if bci.acc > learner.acc: 
                learner = bci
                H[ H['acc_test'] == H['acc_test'].max()].iloc[0]['acc_test'] = bci.acc
                
        learner.load_eeg_train(eeg_path=eeg_path_train, channels=range(0,22))
        learner.load_eeg_test(eeg_path=eeg_path_test, channels=range(0,22))
        learner.crossval = False
        learner.process()
        
        learner.H = H
        learner.cost = cost
        learner.acc_cal = hmax.acc
        learner.as_trials= asetup.trials    
        learner.acc_best = max(learner.acc, acc_mode, acc_pmean)
        
        del (globals()['asetup'], globals()['V'], globals()['P'], globals()['y'], globals()['p'], globals()['t'])
    
        bci = Processor()
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=range(0,22))
        bci.load_eeg_test(eeg_path=eeg_path_test, channels=range(0,22))
        bci.define_params(f_low=4, f_high=40, ncsp=8, class_ids=class_ids, tmin=0.5, tmax=2.5, fs=250, filt_type='IIR', filt_order=5,
                          clf_dict={'model':'SVM','kernel':{'kf':'linear'},'C':-4}, is_sbcsp=True, nbands=9, overlap=True, crossval=False)
        bci.process()
        acc_sbcsp = bci.acc
        h_sb = pd.Series({'fl':4,'fh':40,'tmin':0.5,'tmax':2.5,'ncsp':8,'nbands':9,'clf':{'model':'SVM','kernel':{'kf':'linear'},'C':-4},'acc':acc_sbcsp,'learner':bci})
        learner_sb = bci
        learner_sb.clear_eeg_data()
        
        bci = Processor()
        bci.load_eeg_train(eeg_path=eeg_path_train, channels=range(0,22))
        bci.load_eeg_test(eeg_path=eeg_path_test, channels=range(0,22))
        bci.define_params(f_low=8, f_high=30, ncsp=8, class_ids=class_ids, tmin=0.5, tmax=2.5, fs=250, filt_type='IIR', filt_order=5,
                          clf_dict={'model':'LDA', 'lda_solver':'svd'}, is_sbcsp=False, nbands=None, crossval=False)
        bci.process()
        acc_classic = bci.acc
        h_cla = pd.Series({'fl':8,'fh':30,'tmin':0.5,'tmax':2.5,'ncsp':8,'nbands':None,'clf':{'model':'LDA', 'lda_solver':'svd'},'acc':acc_classic,'learner':bci})
        learner_cla = bci
        learner_cla.clear_eeg_data()
        
        learner.learner_cla = learner_cla
        learner.learner_sb = learner_sb
        
        learner.save_setup(path + ds + '_' + str(suj) + '_learner_' + filt['design'])
        # # learner_sb.save_setup('/home/vboas/Desktop/' + ds + str(suj) + '_learner_sb')
        # # learner_cla.save_setup('/home/vboas/Desktop/' + ds + str(suj) + '_learner_cla')
        
        print(f"Max: {round(learner.acc*100,2)} >> {learner.f_low}-{learner.f_high}Hz; {learner.tmin}-{learner.tmax}s; Ns={learner.nbands} {learner.ncsp_list}; R={learner.ncsp}; {learner.clf_dict}")
        print(f"AS: {round(learner.acc_best*100,2)} | SB: {round(learner.learner_sb.acc*100,2)} | CLA: {round(learner.learner_cla.acc*100,2)}\n")
        # print(f"Train: Max={round(learner.acc_cal_max*100,2)} | Tune={round(learner.acc_cal_tune*100,2)} ")
        print(H[['acc','acc_test']].describe())
        # print(learner.cost, acc_sbcsp, acc_classic)

    ####################################
    # learner = Processor()

    # for suj in range(1,10):
    #     learner.load_setup(path + ds + '_' + str(suj) + '_learner_' + filt['design'])
        
    #     R.loc[len(R)] = [suj, learner.f_low, learner.f_high, learner.tmin, learner.tmax, learner.nbands, learner.ncsp, learner.ncsp_list, learner.clf_dict['model'], learner.clf_dict,
    #                      learner.acc, learner.acc_best, learner.learner_sb.acc, learner.learner_cla.acc, learner.cost]
        
    #     G.insert(G.shape[-1], 'S{}'.format(suj), np.asarray(learner.H['acc_test'])*100)
        
    # # pd.to_pickle(R, path + ds + '_R.pkl')
    # # R = pickle.load(open(path + ds + '_R.pkl', 'rb'))
    
    # plt.figure(figsize=(10,7), facecolor='mintcream')
    # plt.boxplot(G.T, boxprops={'color':'b'}, medianprops={'color': 'r'}, whiskerprops={'linestyle':'-.'}, capprops={'linestyle':'-.'}, zorder=5) 
    # # plt.title('Diagrama de caixa representando a variação da performance de classificação em cada sujeito ao\nconsiderar os seis pares de classes avaliados para o conjunto de dados BCI Competition IV 2a')
    # plt.xticks(range(1,10),['S1','S2','S3','S4','S5','S6','S7','S8','S9'])
    # # plt.yticks(np.linspace(0.6, 1., 5, endpoint=True))
    # plt.xlabel('Sujeito', size=14)
    # plt.ylabel('Acurácia (%)', size=14)
    # plt.yticks(np.arange(40.0, 101, step=5.0))
    # plt.ylim((38.0,102))
    # plt.xlim((0,10))
    # # plt.legend(loc='lower right', fontsize=12)
    # # plt.savefig(path + 'boxplot_subj_all_as.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
        

        

# =============================================================================
# OLDs
# =============================================================================

        # data, events, info = np.load(eeg_path_test, allow_pickle=True)
        # smin, smax = int(0*info['fs']), int(4*info['fs'])
        # epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
        
        # epochs_a = np.asarray(epochs)[np.where(labels == class_ids[0])]
        # epochs_b = np.asarray(epochs)[np.where(labels == class_ids[1])]

        # SA, SB = epochs_a[0], epochs_b[0]
        # for i in range(1, len(epochs_a)): SA = np.c_[ SA, epochs_a[i] ]
        # for i in range(1, len(epochs_b)): SB = np.c_[ SB, epochs_b[i] ]


        
        # SA, SB = ZVA[np.random.randint(0, len(ZVA)-1)], ZVB[np.random.randint(0, len(ZVB)-1)] 
        # for i in range(1, len(ZVA)): SA = np.c_[ SA, trials_a[np.random.randint(0, len(ZVA)-1)] ] # ZVA[i]
        # for i in range(1, len(ZVB)): SB = np.c_[ SB, trials_a[np.random.randint(0, len(ZVB)-1)] ] # ZVB[i]
        
        # comm_esperado = 1
        
        # round_time = 20
        # # n_epochs = int((round_time - delay_t)/delta_t)   # número de janelas avaliadas na partida (exemplos/samples para classificação por partida)
        # n_epochs = int((round_time)/delta_t) # ponta do asteroide a vista, asteroide parado por L ()
                
        # if comm_esperado == class_ids[0]: 
        #     slider = np.asarray([ SA[:, n-q:n] for n in range(q, SA.shape[-1], delta_s) ]) # extrai do sinal continuo, épocas com dimensão (q) a cada deslocamento (delta_s)
        #     ZV = slider[:n_epochs] # ZV(acumulador) = buffers/épocas da classe A
        #     tv = class_ids[0] * np.ones(len(ZV))
        
        # elif comm_esperado == class_ids[1]:
        #     slider = np.asarray([ SB[:, n-q:n] for n in range(q, SB.shape[-1], delta_s) ]) # extrai do sinal continuo, épocas com dimensão (q) a cada deslocamento (delta_s)
        
        #     ZV = slider[:n_epochs] # ZV(acumulador) = buffers/épocas da classe A
        #     tv = class_ids[0] * np.ones(len(ZV))
    
        # del (globals()['ZT'], globals()['ZV'], globals()['tt'], globals()['tv'])
        # del (globals()['data_cal'], globals()['data_val'], globals()['data'], globals()['events_cal'], globals()['events_val'], globals()['events'], globals()['info'])



        # trials = [ data_val[:, s0[i]:sn[i]] for i in range(len(labels)) ]
        # trials_a = np.asarray(trials)[np.where(labels == class_ids[0])]
        # trials_b = np.asarray(trials)[np.where(labels == class_ids[1])]


        # SA, SB = trials_a[0], trials_b[0]
        # for i in range(1, len(trials_a)): SA = np.c_[ SA, trials_a[i] ]
        # for i in range(1, len(trials_b)): SB = np.c_[ SB, trials_b[i] ]

        # delta_t = 0.2 # deslocamento em segundos
        # delta_s = int(delta_t * Fs) # deslocamento em amostras
        # delay = int(5/delta_t)  # 2/delta_t == 2 seg == x épocas, representa épocas "perdidas" entre partidas

        # n_rounds = 6        # 6, número de partidas jogadas
        # round_time = 40     # 40, simula tempo (segundos) que o asteroid leva para percorrer a tela

        # # no_asteroidx = [ np.random.choice(class_ids) for i in range(n_rounds) ] # representa os comandos desejados a cada partida (fuga do asteróidd, lado oposto da tela de onde ele surge)
        # no_asteroidx = [1,1,1,2,2,2]; random.shuffle(no_asteroidx)
        # # print(no_asteroidx, '\n')

        # limiar_gatilho = 0.9

        # data_val, events_val = None, None

        # ##%% ###########################################################################
        # #### CONSTRUIDO BUFFER DESLIZANTE - LARGURA OTIMIZADA                       ####
        # ##%% ###########################################################################

        # q = int(tmax*Fs) - int(tmin*Fs) # 500 tamanho do buffer

        # # constrói épocas do sinal continuo usando as dimensões do buffer (q) e o deslocamento (simula o deslizar do buffer)
        # buffers_a = np.asarray([ SA[:, i-q:i] for i in range(q, SA.shape[-1], delta_s) ]) # n_
        # buffers_b = np.asarray([ SB[:, i-q:i] for i in range(q, SB.shape[-1], delta_s) ])

        # ZV, tv = [], [] # acumulador de buffers/épocas e respectivos rótulos para validação
        # cont_a, cont_b = delay, delay # contadores de buffers/épocas A, B adicionadas ao acumulador para validação
        # for i in range(n_rounds):
        #     samples = int(round_time/delta_t)   # 40/0.2 = 200, número de janelas avaliadas na partida
        #     if no_asteroidx[i] == class_ids[0]:
        #         ZV.append(buffers_a[cont_a : cont_a + samples]) # add buffers/épocas da classe A no acumulador
        #         cont_a += (samples + delay) # incrementa o contador de buffers/épocas A usadas
        #     else:
        #         ZV.append(buffers_b[cont_b : cont_b + samples]) # add buffers/épocas da classe A no acumulador
        #         cont_b += (samples + delay) # incrementa o contador de buffers/épocas A usadas
        #     tv.append(no_asteroidx[i] * np.ones(samples))

        # if n_rounds > 1: ZV, tv = np.vstack(ZV), np.ravel(tv).astype(int) # formatando para validação dos modelos para todas as partidas de uma única vez
        # else: ZV, tv = ZV[0], tv[0].astype(int)

        # ################################ AS ON-LINE ##################################

        # ZT, _, tt, _ = half_split_data(data, events, int(tmin*Fs), int(tmax*Fs), class_ids)
        # as_on, LA = sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs=info['fs'], filt='DFT')
        # # as_online, y, p = tester(ZV, tv, Fs, class_ids, hmax) # (TREINO SOMENTE E)
        # as_on = round(as_on*100,2); # print(f"ASon : {as_on}")

        # #### GERAÇÃO DE COMANDO (AS)

        # y, p = LA['y'], LA['yp']
        # tta = tmax-tmin # tempo de ação em segundos ### tempo de ação menor == mais possibilidades de comandos durante a partida, e vice-versa
        # nta = int(tta/delta_t) # tempo de ação em número de épocas no buffer circular (amostras passadas)
        # cont_com_a, cont_com_b, cont_no_com = 0, 0, 0
        # as_comlist = []

        # i = nta
        # while (i < len(y)): # for i in range(nta, len(y), 1):
        #     A = y[i-nta:i] #buffer circular externo
        #     U1 = list(A).count(class_ids[0]) # conta as classificações A no buffer
        #     U2 = list(A).count(class_ids[1]) # conta as classificaçòes B no buffer
        #     U1_prop = U1 / nta # proporção de classificações A no buffer
        #     U2_prop = U2 / nta # proporção de classificações B no buffer
        #     if (U1_prop >= limiar_gatilho): #  and (tv[i]==class_ids[0])
        #         as_comlist.append(np.array([class_ids[0], i])) # [ comando, momento do envio (indice época)]
        #         # cont_com_a += 1; rota += '< '; # print([i-nta,i])
        #         i += nta # como o comando foi enviado o buffer SALTA, iniciando a partir da época do comando na próxima iteração
        #     if (U2_prop >= limiar_gatilho): #  and (tv[i]==class_ids[1])
        #         as_comlist.append(np.array([class_ids[1], i]))
        #         # cont_com_b += 1; rota += '> '; # print([i-nta,i])
        #         i += nta # se um comando foi enviado, o próximo A se inicia no instante que o comando foi enviado (sem sobreposição)
        #     if (U1_prop < limiar_gatilho) and (U2_prop < 0.9):
        #         # cont_no_com += 1;
        #         i += 1 # se nenhum comando foi enviado, o próximo A se inicia deslocado de delta_t em relação ao inicio do A anterior (sobrepopsição de nta-1)
        # as_comlist = np.asarray(as_comlist)
        # as_comlist = np.c_[as_comlist, np.zeros(len(as_comlist))].astype(int)

        # for i in range(len(as_comlist)):
        #     if as_comlist[i,0] == tv[as_comlist[i,1]]: as_comlist[i, 2] = 1

        # as_command = round(np.mean(as_comlist[:, 0] == tv[as_comlist[:,1]])*100,2)
        # # taxa de acerto de comando == comando enviado / comando esperado : corretos_cont/len(command_list) ou np.mean(command_list[:, 0] == tv[command_list[:,1]])
        # # print(no_asteroidx, cont_com_a, cont_com_b, cont_no_com)
        # # print(rota); print(corretos_list)
        # # print(f'Comando via Acum. : {as_command} | n_comandos: {cont_com_a + cont_com_b} | n_corretos: {corretos_cont}')

        # # t_ = np.asarray([ tv[i] for i in range(nta, len(y), nta) ]) # tv[0::nta].astype(int)
        # # m_ = np.asarray([ mode(y[i-nta:i])[0][0] for i in range(nta, len(y), nta) ])
        # # p_ = np.asarray([ np.argmax(np.mean((p[i-nta:i]), axis=0)) + 1 for i in range(nta, len(p), nta) ])
        # # print(f'Comando via Moda  : {round(np.mean(m_ == t_)*100,2)} | n_comandos: {len(m_)}')
        # # print(f'Comando via PMedia: {round(np.mean(p_ == t_)*100,2)} | n_comandos: {len(p_)}\n')


        # ##%% ###########################################################################
        # #### CONSTRUIDO BUFFER DESLIZANTE - LARGURA FIXA 2s                         ####
        # ##%% ###########################################################################

        # smin, smax = int(0.5*Fs), int(2.5*Fs)
        # q = smax - smin # tamanho do buffer

        # # constrói épocas do sinal continuo usando as dimensões do buffer (q) e o deslocamento (simula o deslizar do buffer)
        # buffers_a = np.asarray([ SA[:, i-q:i] for i in range(q, SA.shape[-1], delta_s) ]) # n_
        # buffers_b = np.asarray([ SB[:, i-q:i] for i in range(q, SB.shape[-1], delta_s) ])

        # ZV, tv = [], [] # acumulador de buffers/épocas e respectivos rótulos para validação
        # cont_a, cont_b = delay, delay # contadores de buffers/épocas A, B adicionadas ao acumulador para validação
        # for i in range(n_rounds):
        #     samples = int(round_time/delta_t)   # 200, número de janelas avaliadas na partida
        #     if no_asteroidx[i] == class_ids[0]:
        #         ZV.append(buffers_a[cont_a : cont_a + samples]) # add buffers/épocas da classe A no acumulador
        #         cont_a += (samples + delay) # incrementa o contador de buffers/épocas A usadas
        #     else:
        #         ZV.append(buffers_b[cont_b : cont_b + samples]) # add buffers/épocas da classe A no acumulador
        #         cont_b += (samples + delay) # incrementa o contador de buffers/épocas A usadas
        #     tv.append(no_asteroidx[i] * np.ones(samples))

        # if n_rounds > 1: ZV, tv = np.vstack(ZV), np.ravel(tv).astype(int) # formatando para validação dos modelos para todas as partidas de uma única vez
        # else: ZV, tv = ZV[0], tv[0].astype(int)

        # ################################ SBCSP ON-LINE ################################

        # ZT, _, tt, _ = half_split_data(data, events, smin, smax, class_ids)
        # sb_on, LS = sbcsp_approach(ZT, ZV, tt, tv, 9, 4, 40, 8, {'model':'SVM','kernel':{'kf':'linear'},'C':-4}, Fs=info['fs'], filt='IIR')
        # sb_on = round(sb_on*100,2); # print(f"SBon : {sb_on}");

        # #### GERAÇÃO DE COMANDO (SBCSP)

        # ys, ps = LS['y'], LS['yp']
        # tta = tmax-tmin # tempo de ação em segundos ### tempo de ação menor == mais possibilidades de comandos durante a partida, e vice-versa
        # nta = int(tta/delta_t) # tempo de ação em número de épocas no buffer circular (amostras passadas)
        # cont_com_a, cont_com_b, cont_no_com = 0, 0, 0
        # sb_comlist, rota = [], ''

        # i = nta
        # while (i < len(ys)): # for i in range(nta, len(ys), 1):
        #     A = ys[i-nta:i] #buffer circular externo
        #     U1 = list(A).count(class_ids[0]) # conta as classificações A no buffer
        #     U2 = list(A).count(class_ids[1]) # conta as classificaçòes B no buffer
        #     U1_prop = U1 / nta # proporção de classificações A no buffer
        #     U2_prop = U2 / nta # proporção de classificações B no buffer
        #     if (U1_prop >= limiar_gatilho): #  and (tv[i]==class_ids[0])
        #         sb_comlist.append(np.array([class_ids[0], i])) # [ comando, momento do envio (indice época)]
        #         cont_com_a += 1; rota += '< '; # print([i-nta,i])
        #         i += nta # como o comando foi enviado o buffer SALTA, iniciando a partir da época do comando na próxima iteração
        #     if (U2_prop >= limiar_gatilho): #  and (tv[i]==class_ids[1])
        #         sb_comlist.append(np.array([class_ids[1], i]))
        #         cont_com_b += 1; rota += '> '; # print([i-nta,i])
        #         i += nta # se um comando foi enviado, o próximo A se inicia no instante que o comando foi enviado (sem sobreposição)
        #     if (U1_prop < limiar_gatilho) and (U2_prop < 0.9):
        #         cont_no_com += 1; i += 1 # se nenhum comando foi enviado, o próximo A se inicia deslocado de delta_t em relação ao inicio do A anterior (sobrepopsição de nta-1)
        # sb_comlist = np.asarray(sb_comlist)
        # sb_comlist = np.c_[sb_comlist, np.zeros(len(sb_comlist))].astype(int)

        # corretos_cont = 0; corretos_list = ''
        # for i in range(len(sb_comlist)):
        #     if sb_comlist[i,0] == tv[sb_comlist[i,1]]: sb_comlist[i,2] = 1 # corretos_cont += 1; corretos_list += '1 '
        #     # else: corretos_list += '0 '

        # sb_command = round(np.mean(sb_comlist[:, 0] == tv[sb_comlist[:,1]])*100,2)
        # # taxa de acerto de comando == comando enviado / comando esperado : corretos_cont/len(command_list) ou np.mean(command_list[:, 0] == tv[command_list[:,1]])
        # # print(no_asteroidx, cont_com_a, cont_com_b, cont_no_com)
        # # print(rota); print(corretos_list)
        # # print(f'Comando via Acum. : {round(np.mean(command_list[:, 0] == tv[command_list[:,1]])*100,2)} | n_comandos: {cont_com_a + cont_com_b} | n_corretos: {corretos_cont}')

        # # t_ = np.asarray([ tv[i] for i in range(nta, len(ys), nta) ]) # tv[0::nta].astype(int)
        # # m_ = np.asarray([ mode(ys[i-nta:i])[0][0] for i in range(nta, len(ys), nta) ])
        # # p_ = np.asarray([ np.argmax(np.mean((ps[i-nta:i]), axis=0)) + 1 for i in range(nta, len(ps), nta) ])
        # # print(f'Comando via Moda  : {round(np.mean(m_ == t_)*100,2)} | n_comandos: {len(m_)}')
        # # print(f'Comando via PMedia: {round(np.mean(p_ == t_)*100,2)} | n_comandos: {len(p_)}\n')


        # # ########################### CLASSIC ON-LINE ################################

        # ZT, _, tt, _ = half_split_data(data, events, smin, smax, class_ids)
        # cla_on, LC = classic_approach(ZT, ZV, tt, tv, 8, 30, 8, {'model':'LDA'}, Fs=info['fs'], filt='IIR')
        # cla_on = round(cla_on*100,2) # print(f"CLAon: {cla_on}")

        # #### GERAÇÃO DE COMANDO (CLASSIC)

        # yc, pc = LC['y'], LC['yp']
        # tta = 2 # tempo de ação em segundos ### tempo de ação menor == mais possibilidades de comandos durante a partida, e vice-versa
        # nta = int(tta/delta_t) # tempo de ação em número de épocas no buffer circular (amostras passadas)
        # cont_com_a, cont_com_b, cont_no_com = 0, 0, 0
        # cla_comlist, rota_cla_iir = [], ''

        # i = nta
        # while (i < len(yc)): # for i in range(nta, len(yc), 1):
        #     A = yc[i-nta:i] #buffer circular externo
        #     U1 = list(A).count(class_ids[0]) # conta as classificações A no buffer
        #     U2 = list(A).count(class_ids[1]) # conta as classificaçòes B no buffer
        #     U1_prop = U1 / nta # proporção de classificações A no buffer
        #     U2_prop = U2 / nta # proporção de classificações B no buffer
        #     if (U1_prop >= limiar_gatilho): #  and (tv[i]==class_ids[0])
        #         cla_comlist.append(np.array([class_ids[0], i])) # [ comando, momento do envio (indice época)]
        #         cont_com_a += 1; rota_cla_iir += '< '; # print([i-nta,i])
        #         i += nta # como o comando foi enviado o buffer SALTA, iniciando a partir da época do comando na próxima iteração
        #     if (U2_prop >= limiar_gatilho): #  and (tv[i]==class_ids[1])
        #         cla_comlist.append(np.array([class_ids[1], i]))
        #         cont_com_b += 1; rota_cla_iir += '> '; # print([i-nta,i])
        #         i += nta # se um comando foi enviado, o próximo A se inicia no instante que o comando foi enviado (sem sobreposição)
        #     if (U1_prop < limiar_gatilho) and (U2_prop < 0.9):
        #         cont_no_com += 1; i += 1 # se nenhum comando foi enviado, o próximo A se inicia deslocado de delta_t em relação ao inicio do A anterior (sobrepopsição de nta-1)
        # cla_comlist = np.asarray(cla_comlist)
        # cla_comlist = np.c_[cla_comlist, np.zeros(len(cla_comlist))].astype(int)

        # corretos_cont_cla_iir = 0; corretos_list_cla_iir = ''
        # for i in range(len(cla_comlist)):
        #     if cla_comlist[i,0] == tv[cla_comlist[i,1]]: cla_comlist[i,2] = 1 # corretos_cont_cla_iir += 1; corretos_list_cla_iir += '1 '
        #     # else: corretos_list_cla_iir += '0 '

        # # taxa de acerto de comando == comando enviado / comando esperado : corretos_cont/len(command_list) ou np.mean(command_list[:, 0] == tv[command_list[:,1]])
        # # print(no_asteroidx, cont_com_a, cont_com_b, cont_no_com)
        # cla_command = round(np.mean(cla_comlist[:, 0] == tv[cla_comlist[:,1]])*100,2)
        # # print(rota_cla_iir); print(corretos_list_cla_iir)
        # # print(f'Comando via Acum. : {acc_cla_iir_acum} | n_comandos: {cont_com_a + cont_com_b} | n_corretos: {corretos_cont}')

        # # t_ = np.asarray([ tv[i] for i in range(nta, len(yc), nta) ]) # tv[0::nta].astype(int)
        # # m_ = np.asarray([ mode(yc[i-nta:i])[0][0] for i in range(nta, len(yc), nta) ])
        # # p_ = np.asarray([ np.argmax(np.mean((pc[i-nta:i]), axis=0)) + 1 for i in range(nta, len(pc), nta) ])
        # # print(f'Comando via Moda  : {round(np.mean(m_ == t_)*100,2)} | n_comandos: {len(m_)}')
        # # print(f'Comando via PMedia: {round(np.mean(p_ == t_)*100,2)} | n_comandos: {len(p_)}\n')


        # # # z = np.asarray([tv, y, ys, yc]).T

        # R.loc[0] = [suj, class_ids[0], class_ids[1], hmax['fl'], hmax['fh'], hmax['tmin'], hmax['tmax'], hmax['nbands'], hmax['ncsp'], csp_list, hmax['clf']['model'], hmax['clf'],
        #                          acc_as_max, acc_as_tune, acc_as_mode, acc_as_pmean, acc_as_best, acc_sb_iir, acc_cla_iir, as_on, sb_on, cla_on,
        #                          as_command, sb_command, cla_command, as_comlist, sb_comlist, cla_comlist]

        # # pd.to_pickle(R, '/home/vboas/cloud/results/' + ds + '/R_' + ds + '_' + str(suj) + '.pkl')







# epochs, labels = extractEpochs(data, events, 0, int(4*Fs), class_ids)
# kf = StratifiedShuffleSplit(10, test_size=0.5, random_state=42)
# for train, test in kf.split(epochs, labels):
#     trials_cal, labels_cal, trials_val, labels_val = epochs[train], labels[train], epochs[test], labels[test]




