#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bci_cp import Processor
from hyperopt.plotting import main_plot_history, main_plot_histogram, main_plot_vars, main_plot_1D_attachment

#%%
# # ====================================
# # Load and View Results (all subjects)
# # ====================================
# RA = []
# for ds in ['IV2a', 'IV2b', 'Lee19']:
#     path = '/home/vboas/cloud/results/as_off/' + ds + '/'
#     R = pd.DataFrame(columns=['subj', 'class_ids', 'nchannels', 'fl', 'fh', 'tmin', 'tmax', 'nbands', 'ncsp', 'csp_list', 'clf', 'clf_details', 
#                               'as_acc_cal', 'as_acc', 'as_acc_max', 'sb_acc_cal', 'sb_acc', 'cla_acc_cal','cla_acc',
#                               'as_mcost','as_mcost_iir'])
#     subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55) if ds == 'Lee19' else ['WL']
#     for suj in subjects:
#         sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
#         sname += str(suj)
#         learner = Processor()
#         learner.load_setup(path + sname + '_learner')
#         R.loc[len(R)] = [sname, [1,2], len(learner.channels), learner.f_low, learner.f_high, learner.tmin, learner.tmax, learner.nbands, learner.ncsp, learner.ncsp_list, learner.clf_dict['model'], learner.clf_dict,
#                          learner.acc_cal, learner.acc_best, learner.acc, learner.learner_sb.acc_cal, learner.learner_sb.acc, learner.learner_cla.acc_cal, learner.learner_cla.acc, 
#                          learner.H['cost'].mean(), learner.learner_iir.H['cost'].mean()]
# #    pd.to_pickle(R, path + 'R_' + ds + '.pkl') 
# #    RA.append(R)
# # pd.to_pickle(pd.concat(RA, ignore_index=True), path + '../RFull.pkl')

#%%
path = '/home/vboas/cloud/results/as_off/'
RF = pd.read_pickle(path + 'RFull.pkl')
RF = RF.loc[:71] # apenas 2A, 2B e LE (exclui LINCE)
clf = pd.read_pickle('/home/vboas/Desktop/res_clf.pkl')

RLE = pd.read_pickle(path + 'Lee19/R_Lee19.pkl')
R2B = pd.read_pickle(path + 'IV2b/R_IV2b.pkl')
R2A = pd.read_pickle(path + 'IV2a/R_IV2a.pkl')
t = RF[['as_acc','sb_acc','cla_acc']] # apenas acurácias

print(t.mean()*100) # acurácias médias nas 3 abordagens 

gain = (RF['as_acc'] - RF['sb_acc']) * 100 # ganho entre as abordagens
print(f'Maior ganho SB->AS: {gain.max()}% (suj {np.argmax(gain)+1})')
print(f'Ganho médio SB->AS: {round(np.mean(gain),2)}% +{round(np.std(gain),1)}')
print('\n')
gain = (RF['as_acc'] - RF['cla_acc']) * 100 # ganho entre as abordagens
print(f'Maior ganho BU->AS: {gain.max()}% (suj {np.argmax(gain)+1})')
print(f'Ganho médio BU->AS: {round(np.mean(gain),2)}% +{round(np.std(gain),1)}')

len(t[t['sb_acc'] < t['as_acc']])/len(t) * 100 # percentual de sujeitos com melhora via auto setup
len(t[t['cla_acc'] < t['as_acc']])/len(t) * 100 # percentual de sujeitos com melhora via auto setup

round(len(RF[RF['sb_acc'] >= 0.7]['sb_acc'])/len(RF) * 100, 2) # Percentual de avaliações com acurácia acima de 70% 

#%%
# =============================================================================
# Correlação e coeficiente de determinação R^2 (todas as N_iter e sujeitos)
# =============================================================================
## Calculo do coeficiente de determinação R^2 (coeficiente de correlação (R) ao quadrado)...
## ... r = np.corrcoef(preditora, resposta)
df = pd.DataFrame(columns=['subj','fl','fh','tmin','tmax','ncsp','nbands']) # f'{suj}'.format(suj)
for suj in range(1,73):
    learner = Processor()
    learner.load_setup('/home/vboas/Desktop/all_learners/l'+str(suj))
    # df.insert(df.shape[-1], f'{suj}', np.asarray(learner.H['acc_test'])*100)
    y = learner.H['acc_test']
    learner.H['nbands'] = pd.to_numeric(learner.H['nbands'].replace([None],'1'))
    learner.H['ncsp'] = pd.to_numeric(learner.H['ncsp'])
    learner.H['fl'] = pd.to_numeric(learner.H['fl'])
    learner.H['fh'] = pd.to_numeric(learner.H['fh'])
    df.loc[len(df)] = [suj, (np.corrcoef(learner.H['fl'], y)[0,1]**2), (np.corrcoef(learner.H['fh'], y)[0,1])**2,
                       (np.corrcoef(learner.H['tmin'], y)[0,1])**2, (np.corrcoef(learner.H['tmax'], y)[0,1])**2, 
                       (np.corrcoef(learner.H['ncsp'], y)[0,1])**2, (np.corrcoef(learner.H['nbands'], y)[0,1])**2]
desc = df.describe()

h = []
for suj in range(1,73):
    learner = Processor()
    learner.load_setup('/home/vboas/Desktop/all_learners/l'+str(suj))
    h.append(learner.H[['fl','fh','tmin','tmax','ncsp','nbands','clf','acc','acc_test','cost']])

h = pd.concat(h, axis=0)
h['nbands'] = pd.to_numeric(h['nbands'].replace([None],'1'))
# col = h['fh'] + h['fl']
xh = np.unique(h['tmax'])
yh = [ np.mean(h[h['tmax'] == i]['acc_test']) for i in xh ]
plt.figure(figsize=(10,6), facecolor='mintcream')
plt.plot(xh, yh, c='b')


#%%
# =============================================================================
# Coeficientes kappa
# =============================================================================
kpa = []
for suj in range(10,19):
    learner = Processor()
    learner.load_setup('/home/vboas/Desktop/all_learners/l'+str(suj))
    kpa.append(learner.kpa)

#%%
# =============================================================================
# ## Distribuição das iterações de convergência
# =============================================================================
am = []
for suj in range(1,73):
    learner = Processor()
    learner.load_setup('/home/vboas/Desktop/all_learners/l'+str(suj))
    am.append(learner.H['acc_test'].argmax())
am = np.asarray(am)
print(f"min={am.min()}, max={am.max()}, media={am.mean()}, dp={am.std()}")
am = am + np.ones(72)
# plt.hist(am, [1,20,40,60,80,100])
  
plt.figure(figsize=(15,6), facecolor='mintcream')
plt.grid(axis='y', **dict(ls='--', alpha=1), zorder=1)
plt.grid(axis='x', **dict(ls='--', alpha=.3), zorder=1)
plt.scatter(range(1, 10), am[:9], facecolors='turquoise', marker='o', s=70, alpha=.9, edgecolors='#12905c', zorder=2, label='2A')
plt.scatter(range(10, 19), am[9:18], facecolors='navajowhite', marker='o', s=70, alpha=.9, edgecolors='#987539', zorder=3, label='2B')
plt.scatter(range(19, 73), am[18:], facecolors='steelblue', marker='o', s=70, alpha=.9, edgecolors='#191958', zorder=4, label='LE')
plt.xticks(np.arange(1,73), rotation = 60, fontsize=11)
plt.yticks([1,20,40,60,80,100], fontsize=12)
plt.ylim((-1,102))
plt.xlim((0,73))
plt.ylabel(r'Iteração $\mathbf{h}^*$', fontsize=13)
plt.xlabel('Modelo AS/sujeito', fontsize=13)
plt.legend(loc='lower right', fontsize=12, ncol=3)
plt.savefig('/home/vboas/Desktop/res_scatter_iter_otimo.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
for ds in ['IV2a', 'IV2b', 'Lee19']: # ['IV2a', 'IV2b', 'Lee19']
    R = pd.read_pickle(path + ds + '/R_' + ds + '.pkl')
    print(f'>>> {ds} <<< ')
    print(R[['as_acc','sb_acc','cla_acc']].mean()*100)
    print('')
# R.loc[0, 'clf_details'] = str({'model':'LDA'})

#%%
# subjects = range(1,10) if ds in ['IV2a','IV2b'] else range(1,55)
# # subjects = [5,6,7,8,9]
# for suj in subjects:
#     sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
#     sname += str(suj)
#     learner = Processor()
#     learner.load_setup(path + sname + '_learner')
#     print('')
#     print(f'>>> {sname} <<< ')
#     print(f"Setup: {learner.f_low}-{learner.f_high}Hz; {learner.tmin}-{learner.tmax}s; Ns={learner.nbands} {learner.ncsp_list}; R={learner.ncsp}; {learner.clf_dict}")
#     print(f"Cost: {round(learner.H['cost'].mean(),2)} (+-{round(learner.H['cost'].std(),2)})  IIR={round(learner.learner_iir.H['cost'].mean(),2)} (+-{round(learner.learner_iir.H['cost'].std(),2)})")
#     print(f'AS Acc: {round(learner.acc_best*100,2)} (max={round(learner.acc*100,2)}) (cal={round(learner.acc_cal*100,2)})')
#     print(f'SB Acc: {round(learner.learner_sb.acc*100,2)} (cal={round(learner.learner_sb.acc_cal*100,2)})')
#     print(f'BU Acc: {round(learner.learner_cla.acc*100,2)} (cal={round(learner.learner_cla.acc_cal*100,2)})')
#     # print(learner.H[['acc','acc_test']].describe())  
# #     print(learner.learner_iir.acc) 

#%%
acc_ref = RF['cla_acc']*100
acc_as = RF['as_acc']*100
plt.figure(figsize=(12,7), facecolor='mintcream')
# plt.title('Gráfico de dispersão do desempenho da classificação individual')
plt.scatter(np.asarray(acc_ref).reshape(-1,1), np.asarray(acc_as).reshape(-1,1), facecolors = 'c', marker = 'o', s=50, alpha=.9, edgecolors='firebrick', zorder=3)
plt.scatter(round(acc_ref.mean(),2), round(acc_as.mean(),2), facecolors = 'dodgerblue', marker = 'o', s=100, alpha=1, edgecolors='darkblue', label='Acurácia Média', zorder=5)
plt.plot(np.linspace(30, 110, 1000), np.linspace(30, 110, 1000), color='dimgray', linewidth=1, linestyle='--', zorder=0)
plt.plot(np.ones(1000)*round(acc_ref.mean(),2), np.linspace(30, round(acc_as.mean(),2), 1000), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 
plt.plot(np.linspace(30, round(acc_ref.mean(),2), 1000), np.ones(1000)*round(acc_as.mean(),2), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 
plt.ylim((38, 102))
plt.xlim((38, 102))
plt.xticks(np.arange(40, 102, 5)) 
plt.yticks(np.arange(40, 102, 5))
plt.legend(loc='lower right', fontsize=12)
plt.ylabel(r'Acurácia $auto$ $setup$', fontsize=14)
plt.xlabel('Acurácia CMBU', fontsize=14)
plt.savefig('/home/vboas/Desktop/scatter_as_bu.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%    
# =============================================================================
# Gráfico Dispersão AS - fig:boxplot_Niter
# =============================================================================
G = pd.DataFrame()
for suj in range(1,10):
    learner = Processor()
    learner.load_setup(path + 'IV2a/A' + str(suj) + '_learner')
    G.insert(G.shape[-1], 'S{}'.format(suj), np.asarray(learner.H['acc_test'])*100)
for suj in range(1,10):
    learner = Processor()
    learner.load_setup(path + 'IV2b/B' + str(suj) + '_learner')
    G.insert(G.shape[-1], 'S{}'.format(suj+9), np.asarray(learner.H['acc_test'])*100) 
for suj in range(1,55):
    learner = Processor()
    learner.load_setup(path + 'Lee19/L' + str(suj) + '_learner')
    G.insert(G.shape[-1], 'S{}'.format(suj+18), np.asarray(learner.H['acc_test'])*100) 

plt.figure(figsize=(10,6), facecolor='mintcream')
plt.grid(axis='y', **dict(ls='--', alpha=0.6))

a = np.asarray(learner.H['acc_test'])*100
# # np.argmax(a)
# # a.max()
# aa = [ a[i] for i in range (10, len(a), 5) ]
# # aa = [ a[i-10:i].mean() for i in range (10, len(a), 5) ]
plt.plot(range(1,73), RF['as_acc'])
plt.scatter(range(1,73), RF['as_acc'])
plt.scatter(range(1,73), RF['sb_acc'], c='r')
plt.scatter(range(1,73), RF['cla_acc'], c='g')

# plt.xticks(np.arange(1, len(aa)+1))

plt.boxplot(G.T, vert=True, showfliers=True, notch=False, patch_artist=True, zorder=3,
            boxprops=dict(color='b', facecolor='white', linewidth=1, linestyle='-', alpha=.9), # hatch='/',   
            medianprops=dict(color='crimson', linewidth=2), 
            whiskerprops=dict(color='b',linestyle='-.', linewidth=None), 
            capprops=dict(color='b',linestyle=None), 
            flierprops=dict(marker='o', markeredgecolor='b') # linestyle='-', markerfacecolor='', markersize='', 
            # sym='o'
            ) 
# plt.title('Diagrama de caixa representando a variação da performance de classificação em cada sujeito ao\nconsiderar os seis pares de classes avaliados para o conjunto de dados BCI Competition IV 2a')
plt.xticks(range(1,73),[''+str(suj) for suj in range(1,73) ], fontsize=11)
plt.yticks(np.arange(40.0, 101, step=5.0), fontsize=12)
plt.xlabel('Sujeito', size=13)
plt.ylabel(r'$AC_{\mathbf{h}_i} \ (\%)$', size=15)
plt.ylim((38.0,102))
plt.xlim((0.5,73.5))
plt.legend(loc='best', fontsize=12, ncol=1, 
            borderaxespad=0.2, framealpha=0.99, labelspacing=0.2, prop=dict(size='11')) # bbox_to_anchor=(-0.1,1.1), 
# plt.savefig(path + 'off_' + ds + '_boxplot_niter_subj.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%    
# =============================================================================
# Boxplot Dataset
# =============================================================================
plt.figure(figsize=(10,6), facecolor='mintcream')
plt.grid(axis='y', **dict(ls='--', alpha=0.6))

# plt.scatter(np.arange(1,4,1), RF.iloc[:9][['as_acc','sb_acc','cla_acc']].max()*100, facecolors = 'dodgerblue', 
#             marker = 'o', s=80, alpha=1, edgecolors='darkblue', label=r'$AS_{ac}$ ($\mathcal{M}_{\mathbf{h}^*}$)', zorder=5)

plt.plot(np.linspace(0.5, 3.5, 100), np.ones(100)*RF.iloc[9:18]['as_acc'].mean()*100, 
          color='r', linewidth=2.5, alpha=.8, linestyle='--', label=r'${\mathrm{AS}_{ac_\mathcal{V}}}^{(2)}$', zorder=2)

plt.plot(np.linspace(0.5, 3.5, 100), np.ones(100)*RF.iloc[9:18]['sb_acc'].mean()*100, 
          color='#e79315', linewidth=2.5, alpha=.8, linestyle='-.', label=r'${\mathrm{CMSB}_{ac_\mathcal{V}}}^{(2)}$', zorder=2)

plt.plot(np.linspace(0.5, 3.5, 100), np.ones(100)*RF.iloc[9:18]['cla_acc'].mean()*100, 
          color='k', linewidth=2.5, alpha=.9, linestyle=':', label=r'${\mathrm{CMBU}_{ac_\mathcal{V}}}^{(2)}$', zorder=2)

# plt.boxplot(RF.iloc[18:][['as_acc','sb_acc','cla_acc']].T*100, vert=True, showfliers=True, notch=False, patch_artist=True, zorder=3,
#             boxprops=dict(color='steelblue', facecolor='steelblue', lw=1, ls='-', alpha=1), # hatch='/',   
#             medianprops=dict(color='white', lw=1.5), 
#             whiskerprops=dict(color='steelblue',ls='-', lw=1), 
#             capprops=dict(color='steelblue',ls=None, lw=None), 
#             flierprops=dict(marker='.', markerfacecolor='steelblue', markersize=5, markeredgecolor='steelblue') # ls='-', markerfacecolor='', markersize='', # sym='o'
#             )

plt.boxplot(RF.iloc[9:18][['as_acc','sb_acc','cla_acc']].T*100, vert=True, showfliers=True, notch=False, patch_artist=True, zorder=3,
            boxprops=dict(color='#a98550', facecolor='navajowhite', lw=1, ls='-', alpha=1), # hatch='/',   
            medianprops=dict(color='#a98550', lw=2), 
            whiskerprops=dict(color='#a98550',ls='-', lw=1), 
            capprops=dict(color='#a98550',ls=None, lw=None), 
            flierprops=dict(marker='.', markerfacecolor='#a98550', markersize=4, markeredgecolor='#a98550') # ls='-', markerfacecolor='', markersize='', # sym='o'
            )

# plt.boxplot(RF.iloc[:9][['as_acc','sb_acc','cla_acc']].T*100, vert=True, showfliers=True, notch=False, patch_artist=True, zorder=3,
#             boxprops=dict(color='#0b4a18', facecolor='turquoise', lw=1, ls='-', alpha=1), # hatch='/',   
#             medianprops=dict(color='#0b4a18', lw=2), 
#             whiskerprops=dict(color='#0b4a18',ls='-', lw=1),
#             capprops=dict(color='#0b4a18',ls=None, lw=None), 
#             flierprops=dict(marker='.', markerfacecolor='#0b4a18', markersize=4, markeredgecolor='#0b4a18') # ls='-', markerfacecolor='', markersize='', # sym='o'
#             ) 

# plt.title('Diagrama de caixa representando a variação da performance de classificação em cada sujeito ao\nconsiderar os seis pares de classes avaliados para o conjunto de dados BCI Competition IV 2a')
# plt.xticks(range(1,9),[ str(suj) for suj in range(1,73) ], fontsize=11, rotation=60)
plt.xticks([1,2,3],['AS','CMSB','CMBU'], fontsize=14)
plt.yticks(np.arange(50.0, 101, step=5.0), fontsize=12)
plt.ylabel(r'Acurácia - $ac_\mathcal{V}$ ($\%$)', size=14)
plt.xlabel(r'$^{(2)}$ média entre os modelos associados aos sujeitos no conjunto 2B', size=11, horizontalalignment='left', x=0)
plt.ylim((48.0,102))
# plt.xlim((0.5,len(range(1,9))+.5))
# plt.grid(axis='x', **dict(ls='--', alpha=0.3))
plt.legend(loc='best', fontsize=14, ncol=1, 
            borderaxespad=0.2, framealpha=0.99, labelspacing=0.2, prop=dict(size='13')) # bbox_to_anchor=(-0.1,1.1), 
plt.savefig('/home/vboas/Desktop/off_boxplot_2B.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%    
# =============================================================================
# Gráfico Dispersão AS  - fig:boxplot_2A_Niter
# =============================================================================
# G = pd.DataFrame()
# for suj in range(1,73):
#     learner = Processor()
#     learner.load_setup('/home/vboas/Desktop/all_learners/l'+str(suj))
#     G.insert(G.shape[-1], '{}'.format(suj), np.asarray(learner.H['acc_test'])*100) 

G1 = pd.read_pickle('/home/vboas/Desktop/acc_all_niter_subj.pkl')

plt.figure(figsize=(20,10.5), facecolor='mintcream')
plt.grid(axis='y', **dict(ls='--', alpha=0.6))

plt.scatter(np.arange(1,73,1), RF['as_acc']*100, facecolors='dodgerblue', marker='o', s=80, alpha=1, 
            edgecolors='darkblue', label=r'$\mathrm{AS}_{ac_\mathcal{V}}$ ($\mathcal{M}_{\mathbf{h}^*}$)', zorder=5)

plt.plot(np.linspace(0.5, 72.5, 100), np.ones(100)*RF['as_acc'].mean()*100, 
          color='r', linewidth=2.5, alpha=.8, linestyle='--', label=r'$\mathrm{AS}_{acm_\mathcal{V}}$', zorder=2)

plt.plot(np.linspace(0.5, 72.5, 100), np.ones(100)*RF['sb_acc'].mean()*100, 
          color='#e79315', linewidth=2.5, alpha=.8, linestyle='-.', label=r'$\mathrm{CMSB}_{acm_\mathcal{V}}$', zorder=2)

plt.plot(np.linspace(0.5, 72.5, 100), np.ones(100)*RF['cla_acc'].mean()*100, 
          color='k', linewidth=2.5, alpha=.9, linestyle=':', label=r'$\mathrm{CMBU}_{acm_\mathcal{V}}$', zorder=2)

plt.boxplot(G1.T, vert=True, showfliers=True, notch=False, patch_artist=True, zorder=3,
            boxprops=dict(color='steelblue', facecolor='steelblue', lw=1, ls='-', alpha=1), # hatch='/',   
            medianprops=dict(color='white', lw=None), 
            whiskerprops=dict(color='steelblue',ls='-', lw=1), 
            capprops=dict(color='steelblue',ls=None, lw=None), 
            flierprops=dict(marker='.', markerfacecolor='steelblue', markersize=5, markeredgecolor='steelblue') # ls='-', markerfacecolor='', markersize='', # sym='o'
            )

plt.boxplot(G1.T.iloc[:18], vert=True, showfliers=True, notch=False, patch_artist=True, zorder=3,
            boxprops=dict(color='#a98550', facecolor='navajowhite', lw=1, ls='-', alpha=1), # hatch='/',   
            medianprops=dict(color='#a98550', lw=None), 
            whiskerprops=dict(color='#a98550',ls='-', lw=1), 
            capprops=dict(color='#a98550',ls=None, lw=None), 
            flierprops=dict(marker='.', markerfacecolor='#a98550', markersize=4, markeredgecolor='#a98550') # ls='-', markerfacecolor='', markersize='', # sym='o'
            )

plt.boxplot(G1.T.iloc[:9], vert=True, showfliers=True, notch=False, patch_artist=True, zorder=3,
            boxprops=dict(color='#61b163', facecolor='turquoise', lw=1, ls='-', alpha=1), # hatch='/',   
            medianprops=dict(color='white', lw=None), 
            whiskerprops=dict(color='#61b163',ls='-', lw=1),
            capprops=dict(color='#61b163',ls=None, lw=None), 
            flierprops=dict(marker='.', markerfacecolor='#13794f', markersize=4, markeredgecolor='#61b163') # ls='-', markerfacecolor='', markersize='', # sym='o'
            ) 

# plt.title('Diagrama de caixa representando a variação da performance de classificação em cada sujeito ao\nconsiderar os seis pares de classes avaliados para o conjunto de dados BCI Competition IV 2a')
plt.xticks(range(1,73),[ str(suj) for suj in range(1,73) ], fontsize=11, rotation=60)
plt.yticks(np.arange(40.0, 101, step=5.0), fontsize=12)
plt.xlabel('Sujeito', size=13)
plt.ylabel(r'Acurácia de $\mathcal{M}_{\mathbf{h}_i}$ ($\%$)', size=14)
plt.ylim((38.0,102))
plt.xlim((0.5,len(range(1,73))+.5))
plt.grid(axis='x', **dict(ls='--', alpha=0.3))
plt.legend(loc='best', fontsize=14, ncol=1, 
            borderaxespad=0.2, framealpha=0.99, labelspacing=0.2, prop=dict(size='13')) # bbox_to_anchor=(-0.1,1.1), 
plt.savefig('/home/vboas/Desktop/off_boxplot_niter_subj.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Janela ótima
# =============================================================================
plt.figure(figsize=(9, 14), facecolor='mintcream')
# plt.title('Ajustes ótimos para os hiperparâmetros associados às propriedades da janela temporal usada \nna extração de épocas para cada um dos 72 sujeitos avaliados com o $auto$ $setup$', fontsize=13)
plt.grid(axis='both', **dict(ls='--', alpha=0.6), zorder=0)
plt.fill_between(np.linspace(0.5, 2.5, 100), np.zeros(100), np.ones(100)*75, color='lemonchiffon', lw=3, alpha=.5, zorder=1)
plt.barh(np.arange(1,73), np.asarray(RF['tmax']-RF['tmin']), left= RF['tmin'], color="lightblue", height=0.5, edgecolor='navy', linewidth=1, zorder=3)
plt.plot(np.zeros(100), np.linspace(0, 74, 100), color='crimson', linewidth=1.5, alpha=.8, linestyle='--', label=r'Dica', zorder=2)
# plt.plot(np.ones(100)*0.5, np.linspace(0, 74, 100), color='crimson', linewidth=1.5, alpha=.8, linestyle='-', label=r'Dica', zorder=0)
# plt.plot(np.ones(100)*0.5, np.linspace(0, 74, 100), color='crimson', linewidth=1.5, alpha=.8, linestyle='-', label=r'Dica', zorder=0)
plt.ylabel('Sujeito', size=14)
plt.xlabel('Tempo (seg)', size=14)
plt.xlim((-0.3,4.2))
plt.ylim((0.3,72.7))
plt.xticks(np.arange(0,4.5,0.5), ['Dica', 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], fontsize=11)
plt.yticks(np.arange(1,73), ['{}'.format(i) for i in range(1,73)], fontsize=11)
plt.savefig('/home/vboas/Desktop/res_window.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Histogramas janelas ótimas
# =============================================================================
plt.rcParams["font.family"] = "cursive"
 
t = RF['tmax'] - RF['tmin']
plt.figure(figsize=(15,5), facecolor='mintcream')
plt.subplot(1,2,1)

plt.subplot(1,2,1) 

x = np.unique(RF['tmin'], return_counts=True)[0]
p = np.unique(RF['tmin'], return_counts=True)[1]/len(RF['tmin'])*100
plt.bar(x,p,width=0.4, zorder=2, color='steelblue', **{'edgecolor':'black', 'linewidth':0}, label='LE')

xba = np.unique(RF.iloc[:18]['tmin'], return_counts=True)[0]
pba = np.unique(RF.iloc[:18]['tmin'], return_counts=True)[1]/len(RF['tmin'])*100
plt.bar(xba,pba,width=0.4, zorder=3, color='navajowhite', **{'edgecolor':'black', 'linewidth':0}, label='2B')

xa = np.unique(RF.iloc[:9]['tmin'], return_counts=True)[0]
pa = np.unique(RF.iloc[:9]['tmin'], return_counts=True)[1]/len(RF['tmin'])*100
plt.bar(xa,pa,width=0.4, zorder=3, color='turquoise', **{'edgecolor':'black', 'linewidth':0}, label='2A')

plt.grid(True, axis='y', **dict(ls='--', alpha=0.6), zorder=1)
plt.ylim((0, 62))
plt.xlabel(r'$J_\mathrm{d}$', fontsize=16, fontdict={'family':'monoscape','color':'k','weight':'normal','size':16})
plt.ylabel('(%)', fontsize=14)
plt.yticks(np.arange(5, 65, 5), fontsize=14)
plt.xticks(fontsize=14)
plt.legend(loc='best')

plt.subplot(1,2,2)

x = np.unique(t, return_counts=True)[0]
p = np.unique(t, return_counts=True)[1]/len(t)*100
plt.bar(x,p,width=0.4, zorder=2, color='steelblue', **{'edgecolor':'black', 'linewidth':0}, label='LE')

xba = np.unique(t[:18], return_counts=True)[0]
pba = np.unique(t[:18], return_counts=True)[1]/len(t)*100
plt.bar(xba,pba,width=0.4, zorder=3, color='navajowhite', **{'edgecolor':'black', 'linewidth':0}, label='2B')

xa = np.unique(t[:9], return_counts=True)[0]
pa = np.unique(t[:9], return_counts=True)[1]/len(t)*100
plt.bar(xa,pa,width=0.4, zorder=3, color='turquoise', **{'edgecolor':'black', 'linewidth':0}, label='2A')

plt.grid(True, axis='y', **dict(ls='--', alpha=0.6), zorder=1)
plt.ylim((0, 32))
plt.xlabel(r'$J_\mathrm{l}$', fontsize=16)
plt.ylabel('(%)', fontsize=14)
plt.yticks(np.arange(5, 35, 5), fontsize=14)
plt.xticks(fontsize=14)
plt.legend(loc='best')

plt.savefig('/home/vboas/Desktop/res_hist_window.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%
# =============================================================================
# Histogramas Frequências
# =============================================================================
plt.rcParams["font.family"] = "cursive"
plt.figure(figsize=(10,5), facecolor='mintcream')
x = [r'$\delta_r$', r'$\theta_r$', r'$\alpha_r$', r'$\beta_r$', r'$\gamma_r$']
plt.bar(x,np.asarray([29,49,70,72,42])/len(RF)*100, width=0.8, zorder=2, color='steelblue', **{'edgecolor':'black', 'linewidth':0}, label='LE')
plt.bar(x,np.asarray([8,14,18,18,12])/len(RF)*100,width=0.8, zorder=3, color='navajowhite', **{'edgecolor':'black', 'linewidth':0}, label='2B')
plt.bar(x,np.asarray([4,5,9,9,6])/len(RF)*100,width=0.8, zorder=3, color='turquoise', **{'edgecolor':'black', 'linewidth':0}, label='2A')
plt.grid(True, axis='y', **dict(ls='--', alpha=0.6), zorder=1)
plt.ylim((0, 107))
plt.xlabel('Ritmos cerebrais', fontsize=16)
plt.ylabel('Ocorrências (%)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(np.arange(10, 105, 10), fontsize=16)
plt.legend(loc='best')
plt.savefig('/home/vboas/Desktop/res_hist_freq.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Proporção classificadores Pizza
# =============================================================================
labels = np.unique(RF['clf'], return_counts=True)[0]
sizes = np.unique(RF['clf'], return_counts=True)[1]
explode = (0.02, 0.02, 0.02, 0.02, 0.02)  # only "explode" the 2nd slice

def func(pct, allvals):
    print(pct)
    absolute = pct/100.*np.sum(allvals)
    return "{:.1f}%\n({:d})".format(pct, int(round(absolute,0)))

fig1, ax1 = plt.subplots(figsize=(10,6), subplot_kw=dict(aspect="equal"))
wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct=lambda pct: func(pct, sizes), 
                                    shadow=False, startangle=90, textprops=dict(fontsize=10, color="k")) # '%1.1f%%'
ax1.legend(wedges, labels, title=r"Espaço de ${\phi}$", loc="center left", bbox_to_anchor=(.8, 0, 0.5, .25), fontsize=12)
plt.setp(autotexts, size=12, weight="bold") # bold
plt.setp(texts, size=12, weight="bold")
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# ax1.set_title("Proporção de ocorrência de classificadores entre as instâncias ótimas de hiperparâmetros\n obtidas com o $auto$ $setup$ para os 72 sujeitos avaliados", fontsize=14)
plt.savefig('/home/vboas/Desktop/clf_pie.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%
# =============================================================================
# Histogramas classificadores
# =============================================================================
plt.rcParams["font.family"] = "cursive"
 
plt.figure(figsize=(15,5), facecolor='mintcream')

# kf, kf_freq = np.unique(clf.loc[clf['clf']=='svm', 'kf'], return_counts=True) # ['linear', 'poly', 'rbf', 'sigmoid']
# reg, reg_freq = np.unique(clf.loc[clf['clf']=='svm', 'c'], return_counts=True)
# metrics, metrics_freq = np.unique(clf.loc[clf['clf']=='knn', 'metric'], return_counts=True)# ['chebyshev', 'euclidean', 'manhattan', 'minkowski']
# neig, neig_freq = np.unique(clf.loc[clf['clf']=='knn', 'neig'], return_counts=True)
# eta, eta_freq = np.unique(clf.loc[clf['clf']=='mlp', 'eta'], return_counts=True)
# neurons, neurons_freq = np.unique(clf.loc[clf['clf']=='mlp', 'neurons'], return_counts=True)

plt.subplot(1,2,1)
labels = np.unique(RF['clf'], return_counts=True)[0]
sizes = np.unique(RF['clf'], return_counts=True)[1]
explode = (0.05, 0.01, 0.01, 0.01, 0.01)  # only "explode" the 2nd slice
cores = ['#C71585', '#FF7F50', '#DB7093', '#E9967A', '#DC143C']
def func(pct, allvals):
    # print(pct)
    absolute = pct/100.*np.sum(allvals)
    return "  {:.1f}%\n  ({:d})".format(pct, int(round(absolute,0)))
wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=cores, autopct=lambda pct: func(pct, sizes), 
                                   shadow=False, startangle=90, textprops=dict(fontsize=9, color="#1f1f42")) # '%1.1f%%'
plt.setp(autotexts, size=11, weight="heavy") # bold
plt.setp(texts, size=11, weight="heavy")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.subplot(1,2,2)
x = ['KNN', 'LDA', 'LR', 'SVM', 'MLP'] # np.unique(RF['clf'], return_counts=True)[0]
p = sorted(np.unique(RF['clf'], return_counts=True)[1]/len(RF)*100, reverse=True)
plt.bar(x,p,width=.8, zorder=2, color='steelblue', **{'edgecolor':'black', 'linewidth':0}, label='LE')
xba = ['KNN', 'SVM', 'LR', 'LDA', 'MLP'] # np.unique(RF.iloc[:18]['clf'], return_counts=True)[0]
pba = sorted(np.unique(RF.iloc[:18]['clf'], return_counts=True)[1]/len(RF)*100, reverse=True)
plt.bar(xba,pba,width=.8, zorder=3, color='navajowhite', **{'edgecolor':'black', 'linewidth':0}, label='2B')
xa = ['KNN', 'SVM', 'LR', 'LDA'] # np.unique(RF.iloc[:9]['clf'], return_counts=True)[0]
pa = sorted(np.unique(RF.iloc[:9]['clf'], return_counts=True)[1]/len(RF)*100, reverse=True)
plt.bar(xa,pa,width=.8, zorder=3, color='turquoise', **{'edgecolor':'black', 'linewidth':0}, label='2A')
plt.grid(True, axis='y', **dict(ls='--', alpha=0.6), zorder=1)
plt.ylim((0, 32))
plt.xlabel(r'Classificadores $(\phi)$', fontsize=12)
plt.ylabel('(%)', fontsize=12)
plt.yticks(np.arange(5, 35, 5), fontsize=12)
plt.xticks(fontsize=13)
plt.legend(loc='best', fontsize=12)

plt.savefig('/home/vboas/Desktop/res_classifiers.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%
# =============================================================================
# Frequencia ótima
# =============================================================================
plt.figure(figsize=(16, 9), facecolor='mintcream') 
plt.grid(axis='y', **dict(ls='--', alpha=0.6))
plt.bar(np.arange(1,73), np.asarray(RF['fh']-RF['fl']), bottom=RF['fl'], color="darkseagreen", width=0.1, edgecolor='darkgreen', linewidth=2, zorder=2)
plt.plot(np.linspace(0, 72, 100), np.ones(100)*30, color='b', linewidth=1.5, alpha=.8, linestyle='dotted', label=r'$F_{u}$ típica', zorder=1)
plt.plot(np.linspace(0, 72, 100), np.ones(100)*8, color='crimson', linewidth=1.5, alpha=.8, linestyle='--', label=r'$F_{l}$ típica', zorder=1)
plt.ylabel('Frequência (Hz)', size=18)
plt.xlabel('Sujeito', size=17)
plt.xlim((0.7,72.3))
plt.ylim((-2,48))
plt.yticks(np.arange(0,47,2), fontsize=14)
plt.xticks(np.arange(1,73), ['{}'.format(i) for i in range(1,73)], fontsize=13, rotation=60)
# ax2.set_xticklabels(['S{}'.format(i) for i in range(1,73)], rotation = 45) # va= 'baseline', ha="right",
plt.legend(fontsize=17)
plt.savefig('/home/vboas/Desktop/off_frequency.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Barras Nsb e Ncsp
# =============================================================================
R1 = RF.rename(columns={'nbands':r'$N_s$', 'ncsp':r'$N_{csp}$'})
ax = R1[[r'$N_s$',r'$N_{csp}$']].plot(figsize=(15,6), kind='bar', width=0.75, label=['Ns','Ncsp'], color=['steelblue','orangered'],zorder=3) #color=cores2, 
ax.grid(True, axis='y', **dict(ls='--', alpha=0.6), zorder=0)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.set_xticklabels(['{}'.format(i) for i in range(1,73)], rotation = 60, fontsize=11) # va= 'baseline', ha="right",
ax.set_yticks(np.linspace(1,27,13, endpoint=False))
ax.set_ylim(0,26)
ax.set_ylabel("Valor da instância ótima", fontsize=13)
ax.set_xlabel("Sujeito", fontsize=13)
ax.legend(fontsize=12)
# ax.set_title("Barras Nsb e Ncsp")
plt.savefig('/home/vboas/Desktop/off_bars_ns_ncsp.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%
# =============================================================================
# Análise de custo FFT x IIR
# =============================================================================
iir, fft = [], []
for suj in range(1,73):
    learner = Processor()
    learner.load_setup('/home/vboas/Desktop/all_learners/l'+str(suj))
    iir.append(learner.learner_iir.H[['nbands','cost']])
    fft.append(learner.H[['nbands','cost']])

fft = pd.concat(fft, axis=0)
iir = pd.concat(iir, axis=0)
fft['nbands'] = pd.to_numeric(fft['nbands'].replace([None],'1'))
iir['nbands'] = pd.to_numeric(iir['nbands'].replace([None],'1'))

nb = np.unique(iir['nbands'])

yfft = [ np.mean(fft[fft['nbands'] == i]['cost']) for i in nb ]
yiir = [ np.mean(iir[iir['nbands'] == i]['cost']) for i in nb ]

plt.figure(figsize=(10,6), facecolor='mintcream')
# plt.plot(nb, yfft, 'o-', c='#1a72a8', lw=3, label='FFT', markerfacecolor='None', ms=10)
plt.plot(nb, yfft, '-', c='#1a72a8', lw=2.5, label='FFT')
plt.plot(nb, yiir, '--', c='#c1840b', lw=2.5, label='IIR')
plt.ylim((0, np.max(yiir)+.3))
plt.xlim((0.5,25.5))
plt.xticks(np.arange(1,26,1), fontsize=11)
plt.yticks(np.linspace(0, 5, 21), fontsize=11)
plt.grid(axis='y', alpha=.5)
plt.xlabel(r'$N_s$', fontsize=13)
plt.ylabel(r'Tempo médio por iteração ($s$)', fontsize=13)
plt.legend(loc='upper left', fontsize=12)
plt.savefig('/home/vboas/Desktop/time_cost_curves.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Dados Tabela FFT x IIR
# =============================================================================
iir = []
for suj in range(1,73):
    learner = Processor()
    learner.load_setup('/home/vboas/Desktop/all_learners/l'+str(suj))
    iir.append(learner.learner_iir.acc)
    
iir, fft = [], []
for suj in range(1,73):
    learner = Processor()
    learner.load_setup('/home/vboas/Desktop/all_learners/l'+str(suj))
    iir.append(learner.learner_iir.H['cost'])
    fft.append(learner.H['cost'])

np.mean(pd.concat(fft, axis=0))
np.mean(pd.concat(iir, axis=0))

#%%
# =============================================================================
# Dispersão comparação correlatos LE
# =============================================================================

lee_csp = [83, 86, 94, 57, 81, 88, 71, 66, 71, 61, 50, 58, 54, 48, 57, 69, 42, 82, 89, 73, 
           100, 85, 68, 54, 57, 44, 70, 97, 98, 66, 57, 97, 89, 47, 52, 94, 81, 52, 52, 58, 
           48, 63, 86, 100, 99, 58, 59, 49, 62, 58, 52, 72, 54, 45]

lee_cssp = [78, 97, 95, 61, 82, 85, 64, 68, 70, 65, 50, 58, 54, 55, 58, 56, 45, 95, 83, 79, 
            100, 92, 57, 66, 59, 44, 62, 99, 98, 65, 57, 99, 92, 45, 54, 94, 95, 53, 49, 56, 
            42, 75, 90, 100, 99, 62, 59, 59, 59, 55, 48, 77, 57, 47]

lee_fbcsp = [84, 99, 94, 53, 84, 89, 71, 84, 70, 54, 48, 50, 54, 53, 60, 63, 54, 93, 89, 82, 
             100, 65, 55, 45, 70, 48, 55, 98, 99, 57, 58, 98, 100, 49, 61, 98, 93, 57, 61, 62, 
             51, 73, 89, 99, 98, 83, 69, 52, 60, 48, 52, 72, 54, 55]

lee_bssfo = [90, 96, 95, 66, 84, 89, 80, 55, 69, 52, 50, 54, 59, 51, 69, 63, 55, 88, 82, 62, 
             100, 90, 53, 51, 86, 48, 51, 99, 98, 55, 58, 99, 100, 55, 58, 100, 93, 52, 81, 64, 
             54, 77, 95, 99, 100, 78, 63, 56, 52, 50, 49, 54, 54, 54]

x = RF.iloc[18:]['as_acc'].sort_values(ascending=True).index

yas = RF.iloc[18:]['as_acc'].sort_values(ascending=True)*100
ysb = np.asarray([ RF.iloc[i]['sb_acc']*100 for i in x])
ybu = np.asarray([ RF.iloc[i]['cla_acc']*100 for i in x])

ylee_csp = np.asarray([ lee_csp[i] for i in x-18])
ylee_cssp = np.asarray([ lee_cssp[i] for i in x-18])
ylee_fbcsp = np.asarray([ lee_fbcsp[i] for i in x-18])
ylee_bssfo = np.asarray([ lee_bssfo[i] for i in x-18])

# plt.scatter(np.arange(1,55,1), RF.iloc[:9][['as_acc','sb_acc','cla_acc']].max()*100, facecolors = 'dodgerblue', 
#             marker = 'o', s=80, alpha=1, edgecolors='darkblue', label=r'$AS_{ac}$ ($\mathcal{M}_{\mathbf{h}^*}$)', zorder=5)

plt.figure(figsize=(15,10), facecolor='mintcream')
plt.plot(np.arange(1,55,1), yas, 'o-', c='r', markeredgecolor='crimson', markersize='8', lw=2.5, label=r'$\mathrm{AS}_{ac_\mathcal{V}}$')
plt.scatter(range(1,55,1), ysb, lw=1, s=70, facecolors='#e79315', marker='^', alpha=1, edgecolors='darkblue', label=r'$\mathrm{CMSB}_{ac_\mathcal{V}}$')
plt.scatter(range(1,55,1), ybu, lw=1.5, s=60, facecolors='b', marker='x', alpha=1, edgecolors='darkblue', label=r'$\mathrm{CMBU}_{ac_\mathcal{V}}$')
# plt.scatter(range(1,55,1), ylee_csp, lw=1, s=70, facecolors='c', marker='*', alpha=1, edgecolors='darkblue', label=r'CSP$^{(1)}$') # Lee et al., 2019
# plt.scatter(range(1,55,1), ylee_cssp, lw=1, s=60, facecolors='b', marker='x', alpha=1, edgecolors='darkblue', label=r'$CSSP^2$') # Lee et al., 2019
# plt.scatter(range(1,55,1), ylee_fbcsp, lw=1, s=70, facecolors='#dde709', marker='^', alpha=1, edgecolors='darkblue', label=r'SBCSP$^{(2)}$') # Lee et al., 2019
# plt.scatter(range(1,55,1), ylee_bssfo, lw=1, s=60, facecolors='#e607f0', marker='x', alpha=1, edgecolors='darkblue', label=r'BSSFO$^{(3)}$') # Lee et al., 2019
plt.ylim((44, 101))
plt.xlim((0.5,54.5))
plt.xticks(np.arange(1,55,1),x+1, fontsize=13, rotation=60)
plt.yticks(np.arange(45, 101, 2.5), fontsize=14)
plt.grid(axis='x', alpha=.5)
plt.grid(axis='y', alpha=.3)
plt.xlabel(r'Sujeito/modelo (ordem crescente de $\mathrm{AS}_{ac_\mathcal{V}}$)', fontsize=16)
# plt.xlabel(r'$^{(1)}$ ... \n $^{(1)}$ ... \n $^{(1)}$ ...', size=11, horizontalalignment='left', x=0)
plt.ylabel(r'Acurácia - $ac_\mathcal{V}$ ($\%$)', fontsize=16)
plt.legend(loc='best', fontsize=15)
plt.savefig('/home/vboas/Desktop/off_curves_approaches_LE.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# # =============================================================================
# # TESTE
# # =============================================================================
# for suj in [1]:#subjects:
#     sname = 'A' if ds=='IV2a' else 'B' if ds=='IV2b' else 'L' if ds=='Lee19' else ''
#     sname += str(suj)
#     learner = Processor()
#     learner.load_setup(path + sname + '_learner')

#     H = learner.H

#     # main_plot_history(trials)
#     # main_plot_histogram(trials)
#     # main_plot_vars(trials, do_show=False)
#     # all_loss = [ trials.trials[i]['result']['loss'] * (-1) for i in range(len(trials.trials)) ]
#     # all_loss1 = pd.Series(all_loss)
#     # idx = np.where(all_loss == np.asarray(all_loss).max())
#     # plt.plot(range(1, len(trials)),all_loss, ls='-', linewidth=1, label='sen(x)')
    
    
#%%   
# # =============================================================================
# # Proporção classificadores 72 sujeitos
# # =============================================================================
# def func(pct, allvals):
#     absolute = int(pct/100.*np.sum(allvals))
#     return "{:.1f}%\n({:d})".format(pct, absolute)

# labels = np.unique(RF['clf'], return_counts=True)[0]
# sizes = np.unique(RF['clf'], return_counts=True)[1]
# explode = (0.02, 0.05, 0.05, 0.05, 0.05)  # only "explode" the 2nd slice

# fig1, ax1 = plt.subplots(figsize=(10,6), subplot_kw=dict(aspect="equal"))
# wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct=lambda pct: func(pct, sizes), shadow=True, startangle=90,
#                                     textprops=dict(fontsize=10, color="k")) # '%1.1f%%'
# ax1.legend(wedges, labels, title=r"Classificadores $({\phi})$", loc="center left", bbox_to_anchor=(.8, 0, 0.5, .25), fontsize=12)
# plt.setp(autotexts, size=12, weight="bold")
# plt.setp(texts, size=14, weight="bold")
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# # ax1.legend()
# ax1.set_title("Proporção de ocorrência de classificadores entre as instâncias ótimas de hiperparâmetros \npara os 9 sujeitos avaliados com o método AS", fontsize=14)
# # plt.savefig(path + '../off_72pie.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

# # =============================================================================
# # PSD
# # =============================================================================
# import math
# from scipy.signal import welch, butter, lfilter
# from bci_utils import extractEpochs, nanCleaner
# from scipy.fftpack import fft

# data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01T.npy', allow_pickle=True)
# Fs = info['fs']
# class_ids = [1,2]
# smin = math.floor(0.5 * Fs)
# smax = math.floor(2.5 * Fs)
# buffer_len = smax - smin
# epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
# epochs = nanCleaner(epochs)

# ch = 13 # 7,13 = hemisf esquerdo (atenua RH) || 11,17 = hemisf direito (atenua LH)
# lado = 'hemif. esquerdo' if ch in [7,13] else 'hemif. direito'

# X = [ epochs[np.where(labels == i)] for i in class_ids ]
# Xa = X[0] # all epochs LH
# Xb = X[1] # all epochs RH

# D = np.eye(22,22) - np.ones((22,22))/22
# Ra = np.asarray([D @ Xa[i] for i in range(len(Xa))])
# Rb = np.asarray([D @ Xb[i] for i in range(len(Xb))])

# b, a = butter(5, [8/125, 30/125], btype='bandpass')
# Ra = lfilter(b, a, Ra)
# Rb = lfilter(b, a, Rb)

# xa = Ra[:,13] # all epochs, 1 channel, all samples LH
# xb = Rb[:,13] # all epochs, 1 channel, all samples RH

# plt.psd(xa, 500, 250, c='r')
# plt.psd(xb, 500, 250, c='b')

# ### Welch
# freq, pa = welch(xa, fs=Fs, nfft=(xa.shape[-1]-1)) # nfft=499 para q=500
# _   , pb = welch(xb, fs=Fs, nfft=(xb.shape[-1]-1)) 
# pa, pb = np.real(pa), np.real(pb)
# ma, mb = np.mean(pa,0), np.mean(pb,0)

# plt.subplots(figsize=(10, 5), facecolor='mintcream')
# plt.subplot(2, 1, 1)
# plt.plot(freq, ma*1e5, label='mão esquerda', c='r')  # np.log10(ma)
# plt.plot(freq, mb*1e5, label='mão direita', c='b')
# plt.xlim((0,40))
# # plt.ylim((-14, -11.5))
# plt.title('Welch')
# plt.ylabel(r'$\mu$V')
# plt.xlabel('Frequência (Hz)')
# plt.legend()

# ### FFT
# T = 1/Fs
# freq = np.fft.fftfreq(xa.shape[-1], T)
# mask = freq>0
# freq = freq[mask]
# fa = np.abs(np.fft.fft(xa))[:, mask]
# fb = np.abs(np.fft.fft(xb))[:, mask]
# ma, mb = np.mean(fa,0), np.mean(fb,0)
# plt.subplot(2, 1, 2)
# plt.plot(freq, ma*1e5, label='LH')
# plt.plot(freq, mb*1e5, label='RH')
# plt.xlim((0,40))
# plt.title('FFT')
# plt.ylabel(r'$\mu$V')
# plt.xlabel('Frequência (Hz)')
# plt.legend()
# # plt.savefig(path + 'off_' + ds + '_psd_welch.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


