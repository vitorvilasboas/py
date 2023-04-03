# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from scripts.bci_utils import BCI, extractEpochs, nanCleaner
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit

if __name__ == '__main__':
    delta_t = 0.5
    W = np.asarray([[i,f,f-i] for i in np.arange(0,3.1,delta_t) for f in np.arange(i+1,4.1,delta_t)])
    
    cores1 = ['olive','firebrick','orange','lime','k','purple','gray','green','c']
    cores2 = ['c','m','darkorange','green','hotpink','firebrick','peru']
    markers = ["P","s","8","d","o","p","X","h","v"]
    
    ## I. Several classifiers -------------------------------------------------
    # clf = [{'model':'LR'},{'model':'Bayes'},{'model':'LDA'},{'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4},
    #         {'model':'KNN', 'metric':'minkowski', 'neig':5},{'model':'DTree', 'crit':'entropy'},
    #         {'model':'MLP', 'eta':-4, 'activ':{'af':'logistic'}, 'n_neurons':100, 'n_hidden':1}]
    # C = {}
    # for c in clf:
    #     A = []
    #     for i in range(1,10):
    #         data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(i) + '.npy', allow_pickle=True)
    #         asuj = []
    #         for w in W:
    #             bci = BCI(data=data, events=events, class_ids=[1, 2], fs=info['fs'], overlap=True, crossval=True, nfolds=10, 
    #                       test_perc=0.2, f_low=8, f_high=30, tmin=w[0], tmax=w[1], ncsp=8, ap={'option':'classic'}, 
    #                       filtering={'design':'IIR', 'iir_order':5}, clf=c)        
    #             bci.evaluate()
    #             asuj.append(bci.acc) # classification results of 10x5-fold cross validation
    #         A.append(asuj)
    #     C[c['model']] = A
    # pickle.dump(C, open('/home/vboas/cloud/overleaf/Artigo_DM/quali_all_clfs_dm.pkl', 'wb'))  
    # R = pd.DataFrame(np.c_[W, np.asarray(C['LDA']).T], columns=['t0','tn','L','S1','S2','S3','S4','S5','S6','S7','S8','S9'])
    # pd.to_pickle(R, '/home/vboas/cloud/overleaf/Artigo_DM/quali_LDA_clfs_dm.pkl')
    
    ### Análise e visualização dos resultados
    # C = pickle.load(open('/home/vboas/cloud/overleaf/Artigo_DM/quali_all_clfs_dm.pkl', 'rb'))
    # clf_models = ['LR','Bayes','LDA','SVM','KNN','DTree','MLP']
    # MC = np.array([[np.mean(C[c]), np.median(C[c]), np.std(C[c])] for c in clf_models])
    # MC = pd.DataFrame(MC, columns=['mean','median','dp']) # Medidas de Centralidade de todos os classificadores
    # MC.insert(0, 'clf', clf_models)
    
    # ### Barras Desempenho médio de todos os classificadores
    # M = [np.asarray([np.median(C[c][i]) for i in range(0,9)]).T for c in clf_models]    
    # M = pd.DataFrame(np.concatenate([M], axis=1).T, columns=['{}'.format(c) for c in clf_models]) # Medianas por sujeito
    # ax = (M*100).plot(figsize=(9,6), kind='bar', width=0.75) #color=cores2, 
    # ax.grid(True, axis='y', **dict(ls='--', alpha=0.6))
    # ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.set_xticklabels(['S{}'.format(i) for i in range(1,10)], rotation = 360) # va= 'baseline', ha="right",
    # ax.set_yticks(np.linspace(40,105,13, endpoint=False))
    # ax.set_ylim(40,100)
    # ax.set_ylabel("Acurácia de Classificação (%)", fontsize=13)
    # ax.set_xlabel("Sujeito", fontsize=13)
    # # ax.set_title("Desempenho Médio dos modelos de classificação por sujeito")
    # plt.savefig('/home/vboas/cloud/overleaf/Artigo_DM/quali_clfs_bars_acc.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    ## ------------------------------------------------------------------------
    
    # # II. LDA only -- Train Windw == Test Window (idem qualificação) ---------
    # A = []
    # for i in range(1,10):
    #     data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(i) + '.npy', allow_pickle=True)
    #     asuj = []
    #     for w in W:
    #         bci = BCI(data=data, events=events, class_ids=[1, 2], fs=info['fs'], overlap=True, crossval=True, nfolds=10, 
    #                   test_perc=0.2, f_low=8, f_high=30, tmin=w[0], tmax=w[1], ncsp=8, ap={'option':'classic'}, 
    #                   filtering={'design':'IIR', 'iir_order':5}, clf={'model':'LDA'})        
    #         bci.evaluate()
    #         asuj.append(bci.acc) # classification results of 10x5-fold cross validation
    #     A.append(asuj)
    # R = pd.DataFrame(np.c_[W, np.asarray(A).T], columns=['t0','tn','L','S1','S2','S3','S4','S5','S6','S7','S8','S9'])
    # pd.to_pickle(R, '/home/vboas/cloud/overleaf/Artigo_DM/quali_LDA_dm.pkl')
    ## ------------------------------------------------------------------------
    
    ## III. Validação Janela Deslizante ---------------------------------------
    # A = []
    # for i in range(1,10):
    #     data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(i) + '.npy', allow_pickle=True)
    #     bci = BCI(class_ids=[1, 2], fs=info['fs'], overlap=True, f_low=8, f_high=30, ncsp=8, ap={'option':'classic'},
    #               filtering={'design':'IIR', 'iir_order':5}, clf={'model':'LDA'})
    #     asuj = []
    #     for w in W:
    #         epochs, labels = extractEpochs(data, events, 0, 1000, [1,2]) # full trial = (-500, 1375)
    #         epochs = nanCleaner(epochs)
    #         cross_scores = []
    #         kf = StratifiedShuffleSplit(10, test_size=0.2, random_state=42)
    #         for train, test in kf.split(epochs, labels):
    #             ZT, tt = epochs[train], labels[train]
    #             ZV0, tv0 = epochs[test], labels[test] 
                
    #             smin, smax = int(w[0]*info['fs']), int(w[1]*info['fs'])
    #             # smin, smax = int(W[6,0]*info['fs']), int(W[6,1]*info['fs'])
    #             ZT = ZT[:,:,smin:smax]
                
    #             delta_s = int(delta_t * info['fs'])
    #             q = 500 # smax - smin  # largura janela deslizante em amostras (TESTAR com os dois)
                
    #             #### Opção 1: vários rótulos para cada trial de validação (1 para cada deslocamento da janela deslizante) (apenas 1 execução da cadeia para todos os trials de validação)
    #             # ZV, tv = [], []
    #             # for trial,cue in zip(ZV0,tv0):
    #             #     n = q   # n = fim janela deslizante
    #             #     while n <= trial.shape[-1]:
    #             #         ZV.append(trial[:, n-q:n]); 
    #             #         tv.append(cue); 
    #             #         n += delta_s
    #             # ZV, tv = np.asarray(ZV), np.asarray(tv)
    #             # acc, _, _ = bci.classic_approach(ZT, ZV, tt, tv)
    #             # cross_scores.append(acc)
                
    #             #### Opção 2: 1 Rótulo por trial de validação == moda na trial (+ lento já que 1 execução da cadeia por trial de validação)
    #             ZV, tv = [], []
    #             for trial,cue in zip(ZV0,tv0):
    #                 n = q   # n = fim janela deslizante
    #                 z,t = [],[]
    #                 while n <= trial.shape[-1]:
    #                     z.append(trial[:, n-q:n]); 
    #                     t.append(cue); 
    #                     n += delta_s
    #                 ZV.append(np.asarray(z))
    #                 tv.append(np.asarray(t))    
    #             y, m = [], []
    #             for i in range(len(ZV)): 
    #                 _, _, l = bci.classic_approach(ZT, ZV[i], tt, tv[i])
    #                 y.append(l['y'])
    #                 m.append(mode(l['y'])[0][0])
    #             acc = np.mean(m == tv0)
    #             cross_scores.append(acc)
            
    #         asuj.append(np.mean(cross_scores))
        
    #     A.append(asuj)
    # R = pd.DataFrame(np.c_[W, np.asarray(A).T], columns=['t0','tn','L','S1','S2','S3','S4','S5','S6','S7','S8','S9'])
    # pd.to_pickle(R, '/home/vboas/cloud/overleaf/Artigo_DM/buf_LDA_dm.pkl')
    ## ------------------------------------------------------------------------
    
    ## Vizualization ----------------------------------------------------------
    # C = pickle.load(open('/home/vboas/cloud/overleaf/Artigo_DM/quali_all_clfs_dm.pkl', 'rb'))
    # R = pd.read_pickle('/home/vboas/cloud/overleaf/Artigo_DM/quali_LDA_clfs_dm.pkl') # I
    R = pd.read_pickle('/home/vboas/cloud/overleaf/Artigo_DM/quali_LDA_dm.pkl')     # II
    # R = pd.read_pickle('/home/vboas/cloud/overleaf/Artigo_DM/buf_LDA_dm.pkl') # III
    
    Lm = np.c_[np.unique(R['L']).T, np.asarray([(R[R['L']==i].iloc[:,3:]).mean() for i in np.unique(R['L'])])]
    Dm = np.c_[np.unique(R['t0']).T, np.asarray([(R[R['t0']==i].iloc[:,3:]).mean() for i in np.unique(R['t0'])])]
    best = np.c_[np.asarray([np.ravel(R[R[i] == R[i].max()][i].iloc[0]) for i in R.columns[3:]]),
                  np.asarray([np.ravel(R[R[i] == R[i].max()].iloc[0,:3]) for i in R.columns[3:]])]
    print(np.mean(R.iloc[:,3:], axis=0)*100)
    print(np.median(R.iloc[:,3:], axis=0)*100)
    print(np.std(R.iloc[:,3:], axis=0)*100)
    M = R.iloc[:,3:].describe()
    
    ### Coeficientes de Correlação (R e R2)
    Dm_r, Dm_r2 = [], []
    Lm_r, Lm_r2 = [], []
    for i in range(1,10):
        lr = LinearRegression()
        lr.fit(np.asarray(R['t0']).reshape(-1,1),R.iloc[:,i+2])
        Dm_r2.append(lr.score(Dm[:,0].reshape(-1,1), Dm[:,i])) # R2 (indep,dep): indica qto localiz explica acc
        Dm_r.append(np.corrcoef(Dm[:,0], Dm[:,i])[0,1])
        # D_r2.append(lr.score(np.asarray(R['t0']).reshape(-1,1), np.asarray(R.iloc[:,i+2])))
        lr.fit(np.asarray(R['L']).reshape(-1,1),R.iloc[:,i+2])
        Lm_r2.append(lr.score(Lm[:,0].reshape(-1,1), Lm[:,i]))
        Lm_r.append(np.corrcoef(Lm[:,0], Lm[:,i])[0,1]) 
    
    ### Boxplot Acurácia LDA por sujeito
    plt.figure(figsize=(9, 6), facecolor='mintcream')
    plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    plt.plot(np.linspace(0.5, 9.5, 100), np.ones(100)*np.mean(np.mean(R.iloc[:,3:], axis=0))*100, 
              color='crimson', linewidth=2, alpha=.8, linestyle='--', label=r'Acurácia Média', zorder=0)
    plt.boxplot(np.asarray(R.iloc[:,3:])*100, vert = True, showfliers = True, notch = False, patch_artist = True, 
                boxprops=dict(facecolor="silver", color="gray", linewidth=1, hatch = '/'))
    plt.xlabel('Sujeito', size=13)
    plt.ylabel('Acurácia de Classificação (%)', size=13)
    plt.ylim((42, 103))
    plt.yticks((np.arange(45, 105, 5)), fontsize=11)
    plt.xticks(np.arange(1,10), ['S{}'.format(i) for i in range(1,10)], fontsize=11)
    plt.legend(loc='lower right')
    # plt.title('Boxplot: Acurácia do classificador LDA por sujeito (MD x ME)')
    # plt.savefig('/home/vboas/cloud/overleaf/Artigo_DM/quali_boxplotLDA_1.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    ### Correlação Localização Janelas e Acurácia Média
    plt.figure(figsize=(9, 6), facecolor='mintcream')
    for i in range(1, 10):
        plt.plot(np.unique(R['t0']), Dm[:,i]*100, color=cores1[i-1], lw=1.5)
        plt.scatter(np.unique(R['t0']), Dm[:,i]*100, color=cores1[i-1], facecolors=cores1[i-1], marker=markers[i-1], label=('S{}' .format(i)))
    plt.plot(np.zeros(100), np.linspace(0, 120, 100), color='crimson', linewidth=2, alpha=.8, linestyle='--', zorder=0)
    plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    # plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
    plt.xlabel('Localização da Janela de EEG (s)', size=13)
    plt.ylabel('Acurácia de Classificação (%)', size=13)
    plt.yscale('linear')
    plt.yticks(np.arange(45, 105, step=5), fontsize=11)
    plt.ylim((47, 105))
    # plt.xticks(np.unique(R['t0']))
    plt.xticks(np.arange(0,3.5,0.5), ['Dica', 0.5, 1, 1.5, 2, 2.5, 3], fontsize=11)
    plt.legend(loc='upper center', ncol=9, fontsize=10, framealpha=0.9, labelspacing=0.2, prop={'size':9}) #borderpad=0.2
    # plt.title('Correlação entre localização da janela e acurácia média LDA')
    # plt.savefig('/home/vboas/cloud/overleaf/Artigo_DM/quali_corr_local_1.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    ### Correlação Largura Janelas e Acurácia Média
    plt.figure(figsize=(9, 6), facecolor='mintcream')
    for i in range(1, 10):
        plt.plot(np.unique(R['L']), Lm[:,i]*100, color=cores1[i-1], lw=1.5)
        plt.scatter(np.unique(R['L']), Lm[:,i]*100, color=cores1[i-1], facecolors=cores1[i-1], marker=markers[i-1], label=('S{}'.format(i)))
    # plt.plot(np.zeros(100), np.linspace(0, 120, 100), color='crimson', linewidth=2, alpha=.8, linestyle='--', zorder=0)
    plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    # plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
    plt.xlabel('Largura da Janela de EEG (s)', size=13)
    plt.ylabel('Acurácia de Classificação (%)', size=13)
    plt.yscale('linear')
    plt.yticks(np.arange(50, 105, step=5), fontsize=11)
    plt.ylim((47, 105))
    plt.xticks(np.unique(R['L']), fontsize=11)
    # plt.xticks(np.arange(0,4.5,0.5), ['Dica', 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], fontsize=12)
    plt.legend(loc='upper center', ncol=9, fontsize=10, framealpha=0.9, labelspacing=0.2, prop={'size':9}) #borderpad=0.2, 
    # plt.title('Correlação entre largura da janela e acurácia média LDA')
    # plt.savefig('/home/vboas/cloud/overleaf/Artigo_DM/quali_corr_largura_1.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    ### Subplot - Correlação Localização/Largura Janelas e Acurácia Média
    plt.subplots(figsize=(20, 6), facecolor='mintcream')
    plt.subplot(1, 2, 1)
    for i in range(1, 10):
        plt.plot(np.unique(R['t0']), Dm[:,i]*100, color=cores1[i-1], lw=1.5)
        plt.scatter(np.unique(R['t0']), Dm[:,i]*100, color=cores1[i-1], facecolors=cores1[i-1], marker=markers[i-1], label=('S{}' .format(i)))
    plt.plot(np.zeros(100), np.linspace(0, 120, 100), color='crimson', linewidth=2, alpha=.8, linestyle='--', zorder=0)
    plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    # plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
    plt.xlabel('Localização da Janela de EEG (s)', size=13)
    plt.ylabel('Acurácia de Classificação (%)', size=13)
    plt.yscale('linear')
    plt.yticks(np.arange(45, 105, step=5), fontsize=11)
    plt.ylim((47, 102))
    plt.xticks(np.arange(0,3.5,0.5), ['Dica', 0.5, 1, 1.5, 2, 2.5, 3], fontsize=11)
    
    plt.subplot(1, 2, 2)
    for i in range(1, 10):
        plt.plot(np.unique(R['L']), Lm[:,i]*100, color=cores1[i-1], lw=1.5)
        plt.scatter(np.unique(R['L']), Lm[:,i]*100, color=cores1[i-1], facecolors=cores1[i-1], marker=markers[i-1], label=('S{}'.format(i)))
    plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    plt.xlabel('Largura da Janela de EEG (s)', size=13)
    plt.ylabel('Acurácia de Classificação (%)', size=13)
    plt.yscale('linear')
    plt.yticks(np.arange(50, 105, step=5), fontsize=11)
    plt.ylim((47, 102))
    plt.xticks(np.unique(R['L']), fontsize=11)
    plt.legend(loc='upper center', bbox_to_anchor=(-0.1,1.1), borderaxespad=0., 
               ncol=9, fontsize=13, framealpha=0.9, labelspacing=0.2, prop={'size':11})
    # plt.savefig('/home/vboas/cloud/overleaf/Artigo_DM/quali_subplot_loc_larg_1.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    ### Boxplot Melhor Janela
    plt.figure(figsize=(9, 6), facecolor='mintcream')
    # plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
    # plt.boxplot(best[:,1:].T, vert=True, showfliers=True, notch=False, patch_artist=True, showmeans=False, meanline=False, medianprops=dict(color='lightblue'),
    #             boxprops=dict(facecolor="lightblue", color="purple", linewidth=1, hatch=None))
    # for i in range(1,10): plt.plot(np.ones(100)*i, np.linspace(best[i-1,1]+0.15,best[i-1,2]-0.15,100), linewidth=27, color="lightblue")
    plt.barh(np.arange(1,10), best[:,3], left=best[:,1], color="lightblue", height=0.5, edgecolor='navy', linewidth=1)
    plt.plot(np.zeros(100), np.linspace(0, 10, 100), color='crimson', linewidth=2, alpha=.8, linestyle='--', label=r'Dica', zorder=0)
    plt.grid(axis='x', **dict(ls='--', alpha=0.6))
    # plt.ylabel('Sujeito', size=14)
    plt.xlabel('Largura e Localização da Janela de EEG (s)', size=13)
    plt.xlim((-0.3,4.2))
    plt.ylim((0.3,9.7))
    plt.xticks(np.arange(0,4.5,0.5), ['Dica', 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], fontsize=11)
    plt.yticks(np.arange(1,10), ['S{}'.format(i) for i in range(1,10)], fontsize=11)
    # plt.legend(loc='best')
    # plt.figure('Figura 9 - Boxplot: Janela com melhor desempenho de generalização por sujeito')
    # plt.savefig('/home/vboas/cloud/overleaf/Artigo_DM/quali_boxplotWIN_1.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    # plt.figure(figsize=(9, 6), facecolor='mintcream')
    # plt.bar(np.arange(1,10), best[:,3], bottom=best[:,1], color="lightblue", width=0.4, edgecolor='navy', linewidth=1)
    # plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    # plt.xlabel('Sujeito', size=14)
    # plt.ylabel('Janela (s)', size=14)
    # plt.ylim((0.3,4.2))
    # plt.yticks((np.arange(0.5,4.5,0.5)), fontsize=12)
    # plt.xticks(np.arange(1,10),fontsize=12)
    # # plt.figure('Figura 9 - Boxplot: Janela com melhor de sempenho de generalização por sujeito')
    # # plt.savefig('/home/vboas/cloud/overleaf/Artigo_DM/boxplotWIN.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    ## ------------------------------------------------------------------------
    
    # for suj in subjects:
    #     data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + '.npy', allow_pickle=True)
    #     bci = BCI(data=data, events=events, class_ids=[1, 2], fs=info['fs'], overlap=True, crossval=False, nfolds=10, 
    #               test_perc=0.5, f_low=8, f_high=30, tmin=best[suj-1,1], tmax=best[suj-1,2], ncsp=8, ap={'option':'classic'}, 
    #               filtering={'design':'IIR', 'iir_order':5}, clf={'model':'LDA'})        
    #     bci.evaluate(); print(bci.acc)
    # del((globals()['data'], globals['events'], globals()['info'], globals()['bci'], globals()['asuj'], globals()['i'], globals()['w']))
    