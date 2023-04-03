# -*- coding: utf-8 -*-
""" Testes usando o space game na plataforma
    CLA, SB, AS = 5 sessões x 10 tentativas(partidas) de controle da nave 
    Taxa acerto comando = media(comandogerado == comandoesperado)
    Calculo ITR usa taxa sucesso controle (area segura alcancada) e tempo agregado conclusão tentativas
        N = len(class_ids) = 2
        bin_rate = (acc_hit * np.log2(acc_hit) + (1-acc_hit) * np.log2((1-acc_hit)/(N-1)) + np.log2(N))
        ITR = nrounds/(full_time/60) * bin_rate

    delta_t = 0.2
    limiar comando = acc validação offline
    tempo de acao (tta) = 2
    overlap check comando = 0.5
    tempo check comando = tta - (tta * overlap)
    qtd epocas classificadas buffer = int((1/delta_t) * tta)
    pausa = 2 + tta (obstáculo visível)
    
    tempo resposta comando (duracao arrasto da nave) = 1 seg 
    nave velocidade vertical (desloc. constante por seg) = 6(10 - tta) px/seg
    nave velocidade horizontal (desloc. por comando) = 100px/seg
    save_targets (posicao da nave no eixo x para objetivo ok) = 150 ou 520 
    
    Dúvidas:
        1. Para calculo do ITR considerar os intervalos entre as tentativas (full_time) ou apenas o tempo útil?
           R: Considerar full_time uma vez que o usuário utiliza a pausa para modular a atividade cerebral 
        2. Adotar como limiar para comando que a predominancia da classe deve ser igual ou superior à:
           x a) acurácia de validacao offline # mostrou-se melhor
             b) um limiar deterministico (exemplo: 0.8)
             c) hibrido (percentual da acurácia de validaćão offline) 
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score as kappa
import matplotlib.pyplot as plt

# # # =============================================================================
# # # ANALYSIS PER APPROACH
# # # =============================================================================
# R = pd.DataFrame(columns=['acc_hit','acc_comm','acc_clf','itr_hit','itr_comm','bit_rate_hit','bit_rate_comm'])
# for ap in ['cla','sb','as']: 
#     # for sess in range(1,2): # sessões de jogo
#     r = pickle.load(open('/home/vboas/cloud/results/as_simu/A1/A1_' + ap + str(1) + '.pkl', 'rb'))
#     acc_val, kpa_val = r['learner'].acc, r['learner'].kpa
#     acc_hit = r['score']/r['nrounds']
#     acc_comm = np.mean(np.asarray(r['comm_sent']) == np.asarray(r['comm_expec']))
#     acc_clf = np.mean(np.asarray(r['y_labels']) == np.asarray(r['targets']))

#     N = 2  # len(r['learner'].class_ids)
#     p = acc_hit if acc_hit < 1 else 0.9999999999999999 ## ITR para round
    
#     lp = p * np.log2(p)
    
#     nlp = (1-p) * np.log2((1-p))
    
#     bit_rate_hit = 1 + lp + nlp
#     if np.isnan(bit_rate_hit): bit_rate_hit = 0
#     itr_hit = (r['nrounds']/(r['full_time']/60)) * bit_rate_hit
    
    
#     print( ap, acc_hit, p, bit_rate_hit, (p * np.log2(p)),  )
    
#     # p = acc_comm  ## ITR para comandos
#     # bit_rate_comm = (p*np.log2(p)) + ((1-p) * np.log2((1-p))) 
#     # if np.isnan(bit_rate_comm): bit_rate_comm = 0
#     # itr_comm = (len(r['comm_sent'])/(r['full_time']/60)) * bit_rate_comm # não pode ser ncheck, por essa medida ser variável
    
    
    
    
#     R.loc[len(R)] = [round(acc_hit*100,2),round(acc_comm*100,2),round(acc_clf*100,2),round(itr_hit,4),round(itr_comm,4),round(bit_rate_hit,4),round(bit_rate_comm,4)]
# # for i in ['acc_hit','acc_comm','acc_clf','itr_hit','itr_comm','bit_rate_hit','bit_rate_comm','p','N','sess']: del globals()[i]; del globals()['i']

#%%
# =============================================================================
# FULL ANALYSIS
# =============================================================================
# ds = 'IV2a' # 'IV2a', 'IV2b', 'Lee19', 'LINCE'
# subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
path = '/home/vboas/cloud/results/as_simu/'
RA = []
for ds in ['IV2a', 'IV2b', 'Lee19']:
    subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
    R = pd.DataFrame(columns=['subj','cfg','session','acc_learner','kpa_learner','score','time','tm_hit','acc_hit',
                              'acc_comm','acc_clf','kpa_comm','kpa_clf','itr_hit','itr_com','n_comm','n_check', 'M_hit', 'M_comm'])
    for ap in ['cla','sb','as']: 
        for suj in subjects:
            sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
            sname += str(suj)
            for sess in range(1,6): # sessões de jogo
                # print(path + sname + '/' + sname + '_' + ap + str(sess) + '.pkl')
                r = pickle.load(open(path + sname + '/' + sname + '_' + ap + str(sess) + '.pkl', 'rb'))
                
                acc_val, kpa_val = r['learner'].acc, r['learner'].kpa
                acc_hit = r['score']/r['nrounds']
                acc_comm = np.mean(np.asarray(r['comm_sent']) == np.asarray(r['comm_expec']))
                acc_clf = np.mean(np.asarray(r['y_labels']) == np.asarray(r['targets']))
                kpa_comm = kappa(np.asarray(r['comm_sent']), np.asarray(r['comm_expec']))
                kpa_clf = kappa(np.asarray(r['y_labels']), np.asarray(r['targets']))
                
                tm_hit = r['full_time']/r['nrounds']
                
                M_hit = r['nrounds'] / (r['full_time']/60)
                M_comm = len(r['comm_sent']) / (r['full_time']/60)
                
                N = 2  # len(r['learner'].class_ids)
                p = acc_hit if acc_hit < 1 else 0.9999999999999999 ## ITR para rounds
                bit_rate = np.log2(N) + (p* np.log2(p)) + (1-p) * np.log2((1-p)/(N-1))
                # if np.isnan(bit_rate): bit_rate = 0.001
                itr_hit = M_hit * bit_rate
                
                p = acc_comm if acc_comm < 1 else 0.9999999999999999  ## ITR para comandos
                bit_rate = np.log2(N) + (p* np.log2(p)) + (1-p) * np.log2((1-p)/(N-1))
                # if np.isnan(bit_rate): bit_rate = 0.001
                itr_comm = M_comm * bit_rate # não pode ser ncheck, por essa medida ser variável
                
                R.loc[len(R)] = [sname, ap.upper(), sess, r['learner'].acc, r['learner'].kpa, r['score'], r['full_time'], tm_hit, acc_hit, 
                                 acc_comm, acc_clf, kpa_comm, kpa_clf, itr_hit, itr_comm, len(r['comm_sent']), r['nchecks'], M_hit, M_comm]   

    pd.to_pickle(R, path + 'Rsimu_' + ds + '.pkl')
    RA.append(R)                                       
RF = pd.concat(RA, ignore_index=True)
pd.to_pickle(RF, path + 'Rsimu_Full.pkl')
# for i in ['acc_val','kpa_val','acc_hit','acc_comm','acc_clf','bit_rate','itr_hit','itr_comm','p','N','r','sname','suj','sess']: del globals()[i]

RF = pd.read_pickle(path + 'Rsimu_Full.pkl')
RA = pd.read_pickle(path + 'Rsimu_IV2a.pkl')
RB = pd.read_pickle(path + 'Rsimu_IV2b.pkl')
RL = pd.read_pickle(path + 'Rsimu_Lee19.pkl')

A, B, S = RF[RF['cfg']=='AS'], RF[RF['cfg']=='CLA'], RF[RF['cfg']=='SB']

print('\n'); print('m t_s')
print('AS:: ', round(A['time'].mean()/60,1), round(A['time'].std()/60,1))
print('SB:: ', round(S['time'].mean()/60,1), round(S['time'].std()/60,1))
print('BU:: ', round(B['time'].mean()/60,1), round(B['time'].std()/60,1))

print('\n'); print('m N_com')
print('AS:: ', A['n_comm'].mean(), A['n_comm'].std())
print('SB:: ', S['n_comm'].mean(), S['n_comm'].std())
print('BU:: ', B['n_comm'].mean(), B['n_comm'].std())

print('\n'); print('m N_xek')
print('AS:: ', A['n_check'].mean(), A['n_check'].std())
print('SB:: ', S['n_check'].mean(), S['n_check'].std())
print('BU:: ', B['n_check'].mean(), B['n_check'].std())

print('\n'); print('m ac clf')
print('AS:: ', round(A['acc_clf'].mean()*100,1), round(A['acc_clf'].std()*100,1))
print('SB:: ', round(S['acc_clf'].mean()*100,1), round(S['acc_clf'].std()*100,1))
print('BU:: ', round(B['acc_clf'].mean()*100,1), round(B['acc_clf'].std()*100,1))

print('\n'); print('m ac hit')
print('AS:: ', round(A['acc_hit'].mean()*100,1), round(A['acc_hit'].std()*100,1))
print('SB:: ', round(S['acc_hit'].mean()*100,1), round(S['acc_hit'].std()*100,1))
print('BU:: ', round(B['acc_hit'].mean()*100,1), round(B['acc_hit'].std()*100,1))

print('\n'); print('m ac com')
print('AS:: ', round(A['acc_comm'].mean()*100,1), round(A['acc_comm'].std()*100,1))
print('SB:: ', round(S['acc_comm'].mean()*100,1), round(S['acc_comm'].std()*100,1))
print('BU:: ', round(B['acc_comm'].mean()*100,1), round(B['acc_comm'].std()*100,1))

print('\n'); print('m kpa com')
print('AS:: ', round(A['kpa_comm'].mean(),3), round(A['kpa_comm'].std(),3))
print('SB:: ', round(S['kpa_comm'].mean(),3), round(S['kpa_comm'].std(),3))
print('BU:: ', round(B['kpa_comm'].mean(),3), round(B['kpa_comm'].std(),3))

print('\n'); print('m M hit')
print('AS:: ', round(A['M_hit'].mean(),2), round(A['M_hit'].std(),1))
print('SB:: ', round(S['M_hit'].mean(),2), round(S['M_hit'].std(),1))
print('BU:: ', round(B['M_hit'].mean(),2), round(B['M_hit'].std(),1))

print('\n'); print('m M com')
print('AS:: ', round(A['M_comm'].mean(),2), round(A['M_comm'].std(),1))
print('SB:: ', round(S['M_comm'].mean(),2), round(S['M_comm'].std(),1))
print('BU:: ', round(B['M_comm'].mean(),2), round(B['M_comm'].std(),1))

print('\n'); print('m ITR hit')
print('AS:: ', round(A['itr_hit'].mean(),1), round(A['itr_hit'].std(),1))
print('SB:: ', round(S['itr_hit'].mean(),1), round(S['itr_hit'].std(),1))
print('BU:: ', round(B['itr_hit'].mean(),1), round(B['itr_hit'].std(),1))

print('\n'); print('m ITR com')
print('AS:: ', round(A['itr_com'].mean(),1), round(A['itr_com'].std(),1))
print('SB:: ', round(S['itr_com'].mean(),1), round(S['itr_com'].std(),1))
print('BU:: ', round(B['itr_com'].mean(),1), round(B['itr_com'].std(),1))


RFM = pd.DataFrame(columns=['subj','cfg','acc_learner','kpa_learner',
                            'mscore','dpscore','mtime','dptime','mtm_hit','dptm_hit','macc_hit','dpacc_hit',
                            'macc_comm','dpacc_comm','macc_clf','dpacc_clf','mkpa_comm','dpkpa_comm',
                            'mkpa_clf','dpkpa_clf','mitr_hit','dpitr_hit','mitr_com','dpitr_com'])
for ds in ['IV2a', 'IV2b', 'Lee19']:
    subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55)
    for ap in ['cla','sb','as']: 
        for suj in subjects:
            sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
            sname += str(suj)
            rm = RF[(RF['cfg']==ap.upper())][RF['subj']==sname]
            RFM.loc[len(RFM)] = [sname, ap.upper(), rm['acc_learner'].mean()*100, rm['kpa_learner'].mean(), 
                                 rm['score'].mean(), rm['score'].std(), rm['time'].mean(), rm['time'].std(), rm['tm_hit'].mean(), rm['tm_hit'].std(), (rm['acc_hit']*100).mean(), (rm['acc_hit']*100).std(),
                                 (rm['acc_comm']*100).mean(), (rm['acc_comm']*100).std(), (rm['acc_clf']*100).mean(), (rm['acc_clf']*100).std(), rm['kpa_comm'].mean(), rm['kpa_comm'].std(),
                                 rm['kpa_clf'].mean(), rm['kpa_clf'].std(), rm['itr_hit'].mean(), rm['itr_hit'].std(), rm['itr_com'].mean(), rm['itr_com'].std()]

K = pd.DataFrame(columns=['subj','gain_sb_m', 'gain_sb_dp', 'gain_bu_m', 'gain_bu_dp', 
                          'as_itr_hit_m', 'as_itr_hit_dp', 'as_itr_com_m', 'as_itr_com_dp',
                          'as_acc_hit_m', 'as_acc_hit_dp', 'as_acc_com_m', 'as_acc_com_dp'])
for ds in ['IV2a', 'IV2b', 'Lee19']:
    subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55)
    for suj in subjects:
        sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
        sname += str(suj)
        ias = RF[(RF['cfg']=='AS')][RF['subj']==sname][['acc_hit','acc_comm','itr_hit','itr_com']]
        isb = RF[(RF['cfg']=='SB')][RF['subj']==sname][['acc_hit','acc_comm','itr_hit','itr_com']]
        ibu = RF[(RF['cfg']=='CLA')][RF['subj']==sname][['acc_hit','acc_comm','itr_hit','itr_com']]
        
        ganho_sb = np.asarray(ias['itr_hit']) - np.asarray(isb['itr_hit'])
        ganho_bu = np.asarray(ias['itr_hit']) - np.asarray(ibu['itr_hit'])
        
        # ganho = ias['itr_hit'].mean() - isb['itr_hit'].mean()
        # ganho2 = ias['itr_hit'].mean() - ibu['itr_hit'].mean()
        
        K.loc[len(K)] = [sname, ganho_sb.mean(), ganho_sb.std(), ganho_bu.mean(), ganho_bu.std(), 
                         ias['itr_hit'].mean(), ias['itr_hit'].std(), ias['itr_com'].mean(), ias['itr_com'].std(),
                         ias['acc_hit'].mean(), ias['acc_hit'].std(), ias['acc_comm'].mean(), ias['acc_comm'].std()]


#%%
isb = np.asarray(RFM[RFM['cfg'] == 'SB']['mitr_com'])
ibu = np.asarray(RFM[RFM['cfg'] == 'CLA']['mitr_com'])
ias = np.asarray(RFM[RFM['cfg'] == 'AS']['mitr_com'])
plt.figure(figsize=(10,6), facecolor='mintcream')
plt.scatter(isb.reshape(-1,1), ias.reshape(-1,1), facecolors = 'orange', marker = '^', s=80, alpha=.9, edgecolors='darkblue', label=r'AS $vs$. CMSB', zorder=3)
plt.scatter(ibu.reshape(-1,1), ias.reshape(-1,1), facecolors = 'b', marker = 'x', s=80, alpha=.9, edgecolors='darkblue', label=r'AS $vs$. CMBU', zorder=3)
plt.plot(np.linspace(-5, 30, 1000), np.linspace(-5, 30, 1000), color='dimgray', linewidth=1, linestyle='--', zorder=0)
# dodgerblue, firebrick
plt.scatter(round(isb.mean(),2), round(ias.mean(),2), facecolors = 'g', marker = 'o', s=100, alpha=1, edgecolors='darkblue', label='Média suj. (CMSB)', zorder=5)
plt.scatter(round(ibu.mean(),2), round(ias.mean(),2), facecolors = 'r', marker = 'o', s=100, alpha=1, edgecolors='k', label='Média suj. (CMBU)', zorder=5)

plt.plot(np.ones(1000)*round(ibu.mean(),2), np.linspace(-2, round(ias.mean(),2), 1000), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 
plt.plot(np.linspace(-2, round(ibu.mean(),2), 1000), np.ones(1000)*round(ias.mean(),2), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 

plt.plot(np.ones(1000)*round(isb.mean(),2), np.linspace(-2, round(ias.mean(),2), 1000), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 
plt.plot(np.linspace(-2, round(isb.mean(),2), 1000), np.ones(1000)*round(ias.mean(),2), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 

plt.ylim((-2, 27))
plt.xlim((-2, 27))
plt.xticks(np.arange(0, 27, 2), fontsize=13) 
plt.yticks(np.arange(0, 27, 2), fontsize=13)
plt.legend(loc='lower right', fontsize=12)
plt.ylabel(r'$ITRm_{com}$  AS', fontsize=14)
plt.xlabel(r'$ITRm_{com}$  CMBU / CMSB', fontsize=14)
plt.savefig('/home/vboas/Desktop/scatter_itr_com.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

# ----------------------------

isb = np.asarray(RFM[RFM['cfg'] == 'SB']['mitr_hit'])
ibu = np.asarray(RFM[RFM['cfg'] == 'CLA']['mitr_hit'])
ias = np.asarray(RFM[RFM['cfg'] == 'AS']['mitr_hit'])
plt.figure(figsize=(10,6), facecolor='mintcream')
plt.scatter(isb.reshape(-1,1), ias.reshape(-1,1), facecolors = 'orange', marker = '^', s=80, alpha=.9, edgecolors='firebrick', label=r'AS $vs$. CMSB', zorder=3)
plt.scatter(ibu.reshape(-1,1), ias.reshape(-1,1), facecolors = 'b', marker = 'x', s=80, alpha=.9, edgecolors='firebrick', label=r'AS $vs$. CMBU', zorder=3)
plt.plot(np.linspace(-1, 6, 1000), np.linspace(-1, 6, 1000), color='dimgray', linewidth=1, linestyle='--', zorder=0)

plt.scatter(round(isb.mean(),2), round(ias.mean(),2), facecolors = 'g', marker = 'o', s=100, alpha=1, edgecolors='darkblue', label='Média suj. (CMSB)', zorder=5)
plt.scatter(round(ibu.mean(),2), round(ias.mean(),2), facecolors = 'r', marker = 'o', s=100, alpha=1, edgecolors='k', label='Média suj. (CMBU)', zorder=5)

plt.plot(np.ones(1000)*round(ibu.mean(),2), np.linspace(-1, round(ias.mean(),2), 1000), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 
plt.plot(np.linspace(-1, round(ibu.mean(),2), 1000), np.ones(1000)*round(ias.mean(),2), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 

plt.plot(np.ones(1000)*round(isb.mean(),2), np.linspace(-1, round(ias.mean(),2), 1000), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 
plt.plot(np.linspace(-1, round(isb.mean(),2), 1000), np.ones(1000)*round(ias.mean(),2), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 

plt.ylim((-.2, 5.2))
plt.xlim((-.2, 5.2))
plt.xticks(np.arange(0, 5.5, .5), fontsize=13) 
plt.yticks(np.arange(0, 5.5, .5), fontsize=13)
plt.legend(loc='lower right', fontsize=12)
plt.ylabel(r'$ITRm_{hit}$  AS', fontsize=14)
plt.xlabel(r'$ITRm_{hit}$  CMBU / CMSB', fontsize=14)
plt.savefig('/home/vboas/Desktop/scatter_itr_hit.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%


# # # =============================================================================
# # # PRINTS
# # # =============================================================================
# # for ds in ['IV2a', 'IV2b', 'Lee19']:
# #     subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
# #     R = pd.read_pickle('/home/vboas/cloud/results/as_simu/Rsimu_' + ds + '.pkl')
# #     acc, itr, tempo = 'acc_hit', 'itr_hit', 'full_time'
# #     for suj in subjects:
# #         sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
# #         sname += str(suj) 
# #         a = R[(R['approach'] == 'AS') & (R['subj'] == sname)]
# #         b = R[(R['approach'] == 'CLA') & (R['subj'] == sname)]
# #         s = R[(R['approach'] == 'SB') & (R['subj'] == sname)]
# #         print(f"{sname} AS :: acc={round(a[acc].mean()*100,2)}+-{round(a[acc].std()*100,1)}; time={round(a[tempo].mean()/60,2)}+-{round(a[tempo].std()/60,1)}; itr={round(a[itr].mean(),2)}+-{round(a[itr].std(),1)}")
# #         print(f"{sname} BU :: acc={round(b[acc].mean()*100,2)}+-{round(b[acc].std()*100,1)}; time={round(b[tempo].mean()/60,2)}+-{round(b[tempo].std()/60,1)}; itr={round(b[itr].mean(),2)}+-{round(b[itr].std(),1)}")
# #         print(f"{sname} SB :: acc={round(s[acc].mean()*100,2)}+-{round(s[acc].std()*100,1)}; time={round(s[tempo].mean()/60,2)}+-{round(s[tempo].std()/60,1)}; itr={round(s[itr].mean(),2)}+-{round(s[itr].std(),1)}")
# #         print('')
# #     for i in ['a','b','s','sname','suj']: del globals()[i]   

# for ds in ['IV2a', 'IV2b', 'Lee19', 'Full']:
#     R = pd.read_pickle('/home/vboas/cloud/results/as_simu/Rsimu_' + ds + '.pkl')
#     acc, itr, tempo = 'acc_hit', 'itr_hit', 'full_time'
#     print(f'>>> {ds} <<<')
#     for i in ['AS','SB','CLA']:
#         acc_md = round(R[(R['approach'] == i)][acc].mean()*100,2)
#         acc_dp = round(R[(R['approach'] == i)][acc].std()*100,1)
#         time_md = round(R[(R['approach'] == i)][tempo].mean()/60,2)
#         time_dp = round(R[(R['approach'] == i)][tempo].std()/60,1)
#         itr_md = round(R[(R['approach'] == i)][itr].mean(),2)
#         itr_dp = round(R[(R['approach'] == i)][itr].std(),1)
#         print(f"M {i} :: acc={acc_md}+-{acc_dp}; time={time_md}+-{time_dp}; itr={itr_md}+-{itr_dp}")
#     print('')
# for i in ['acc_md','acc_dp','time_md','time_dp','itr_md','itr_dp']: del globals()[i]
    
# # print(f"########## {'CLASSIC' if ap=='cla' else 'SBCSP' if ap=='sb' else 'AUTO SETUP'} ##########")
# # print(f"Acc val   = {round(acc_val*100,2)}% (kappa={round(kpa_val,3)})")
# # print(f"Acc hit   = {round(np.mean(acc_hit)*100,2)}% +-{round(np.std(acc_hit)*100,2)}")
# # print(f"Acc comm  = {round(np.mean(acc_comm)*100,2)}% +-{round(np.std(acc_comm)*100,2)}")
# # print(f"Acc clf   = {round(np.mean(acc_clf)*100,2)}% +-{round(np.std(acc_clf)*100,2)}")
# # print(f"ITR Medio = {round(np.mean(ITR),3)} +-{round(np.std(ITR),2)} (bits/min)")
# # print(f"ITR Max   = {round(np.max(ITR),3)} (bits/min)")

# R = pd.read_pickle(path + 'Rsimu_IV2a.pkl')

# RF = pd.read_pickle(path + 'Rsimu_Full.pkl')

# S = R[(R['approach'] == 'AS') & (R['subj'] == 'B1')]
# S.describe()











# # =============================================================================
# # FULL ANALYSIS
# # =============================================================================
# # ds = 'IV2a' # 'IV2a', 'IV2b', 'Lee19', 'LINCE'
# # subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
# path = '/home/vboas/cloud/results/as_simu/'
# RA = []
# for ds in ['IV2a', 'IV2b', 'Lee19']:
#     subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
#     R = pd.DataFrame(columns=['subj','approach','session','acc_val','kpa_val','nrounds','scores','nchecks',
#                               'full_time','util_time','acc_hit','acc_comm','acc_clf','itr_hit','itr_comm'])
#     for ap in ['cla','sb','as']: 
#         for suj in subjects:
#             sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
#             sname += str(suj)
#             for sess in range(1,6): # sessões de jogo
#                 # print(path + sname + '/' + sname + '_' + ap + str(sess) + '.pkl')
#                 r = pickle.load(open(path + sname + '/' + sname + '_' + ap + str(sess) + '.pkl', 'rb'))
#                 acc_val, kpa_val = r['learner'].acc, r['learner'].kpa
#                 acc_hit = r['score']/r['nrounds']
#                 acc_comm = np.mean(np.asarray(r['comm_sent']) == np.asarray(r['comm_expec']))
#                 acc_clf = np.mean(np.asarray(r['y_labels']) == np.asarray(r['targets']))
            
#                 N = 2  # len(r['learner'].class_ids)
#                 p = acc_hit   ## ITR para rounds
#                 bit_rate = (p * np.log2(p) + (1-p) * np.log2((1-p)/(N-1)) + np.log2(N))
#                 if np.isnan(bit_rate): bit_rate = 1
#                 itr_hit = (r['nrounds']/(r['full_time']/60)) * bit_rate
                
#                 p = acc_comm  ## ITR para comandos
#                 bit_rate = (p * np.log2(p) + (1-p) * np.log2((1-p)/(N-1)) + np.log2(N))
#                 if np.isnan(bit_rate): bit_rate = 1
#                 itr_comm = (r['nrounds']/(r['full_time']/60)) * bit_rate # não pode ser ncheck, por essa medida ser variável
                
#                 R.loc[len(R)] = [sname, ap.upper(), sess, r['learner'].acc, r['learner'].kpa, r['nrounds'], r['score'], r['nchecks'],
#                                   r['full_time'],r['util_time'], acc_hit, acc_comm, acc_clf, itr_hit,itr_comm]   

# #    pd.to_pickle(R, path + 'Rsimu_' + ds + '.pkl')
# #    RA.append(R)                                       
# # RFull = pd.concat(RA, ignore_index=True)
# # pd.to_pickle(RFull, path + 'Rsimu_Full.pkl')
# for i in ['acc_val','kpa_val','acc_hit','acc_comm','acc_clf','bit_rate','itr_hit','itr_comm','p','N','r','sname','suj','sess']: del globals()[i]
