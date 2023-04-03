# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 08:07:01 2021
@author: Vitor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# A1 = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/R1/res_as.pkl')
# B1 = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/res_bu_001.pkl') # DFT
# S1 = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/res_sb_001.pkl') # DFT

BU = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/res_bu_002.pkl') # IIR
SB = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/res_sb_002.pkl') # IIR

BU = BU.drop(['exec', 'setup'], axis=1)
SB = SB.drop(['exec', 'setup'], axis=1)

# L = pd.read_pickle("G:/Meu Drive/devs/BCI/results/sbrt20/master_cv5/RESULTS_9.pkl")
# L1 = L[['subj','A','B','as_best','cla_iir','cla_dft']].copy()

# R = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/res_as_003.pkl')

# A = pd.DataFrame(columns=['suj', 'classes', 'media', 'dp'])
# for classes in ['1 2', '1 3', '1 4', '2 3', '2 4', '3 4']:
#     for suj in range(1, 10):
#         media = R[R.suj == suj][R.classes == classes].acc.mean()
#         dp = R[R.suj == suj][R.classes == classes].acc.std()
#         A.loc[len(A)] = [suj, classes, round(media*100,3), round(dp*100,3)]
#     # A.loc[len(A)] = [None, None, None, None]

R1 = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/R1/res_as.pkl')
R2 = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/R2/res_as.pkl')
R3 = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/R3/res_as.pkl')
R4 = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/R4/res_as.pkl')
R = pd.DataFrame(columns=['suj', 'classes', 'exec', 'acc', 'setup'])
for classes in ['1 2', '1 3', '1 4', '2 3', '2 4', '3 4']:
    for suj in range(1, 10):
        m = [R1[R1.suj == suj][R1.classes == classes].acc.mean(), 
             R2[R2.suj == suj][R2.classes == classes].acc.mean(), 
             R3[R3.suj == suj][R3.classes == classes].acc.mean(), 
             R4[R4.suj == suj][R4.classes == classes].acc.mean()]
        # dp = R1[R1.suj == suj][R1.classes == classes].acc.std() if np.argmax(m) == 0 else R2[R2.suj == suj][R2.classes == classes].acc.std() if np.argmax(m) == 1 else R3[R3.suj == suj][R3.classes == classes].acc.std() if np.argmax(m) == 2 else R4[R4.suj == suj][R4.classes == classes].acc.std()

        if   np.argmax(m) == 0: R = R.append(R1[R1.suj == suj][R1.classes == classes], ignore_index=True) 
        elif np.argmax(m) == 1: R = R.append(R2[R2.suj == suj][R2.classes == classes], ignore_index=True)  
        elif np.argmax(m) == 2: R = R.append(R3[R3.suj == suj][R3.classes == classes], ignore_index=True)
        else:                   R = R.append(R4[R4.suj == suj][R4.classes == classes], ignore_index=True)

r = pd.DataFrame(columns=['suj', 'classes', 'media', 'dp'])
for classes in ['1 2', '1 3', '1 4', '2 3', '2 4', '3 4']:
    for suj in range(1, 10):
        r.loc[len(r)] = [suj, classes, round(R[R.suj == suj][R.classes == classes].acc.mean()*100,3), round(R[R.suj == suj][R.classes == classes].acc.std()*100,3)]
    
# print(r.media.mean())

S = pd.DataFrame(columns=['suj', 'classes', 'exec', 'acc', 'fl', 'fu', 'tl', 'tu', 'nf', 'nb', 'clf'])
for i in range(len(R)): 
    S.loc[len(S)] = [R.iloc[i].suj, R.iloc[i].classes, R.iloc[i].exec, R.iloc[i].acc, R.iloc[i].setup[0], 
                     R.iloc[i].setup[1], R.iloc[i].setup[2], R.iloc[i].setup[3], R.iloc[i].setup[4], 
                     R.iloc[i].setup[5], R.iloc[i].setup[6]['model']]

# pd.to_pickle(S, 'G:/Meu Drive/devs/BCI/results/trials/res_as.pkl') 
# S.to_excel('G:/Meu Drive/devs/BCI/results/trials/res_as.xlsx')

# pd.to_pickle(r, 'G:/Meu Drive/devs/BCI/results/trials/res_acc_as_novo.pkl')
r = pd.read_pickle('G:/Meu Drive/devs/BCI/results/trials/res_acc_as_novo.pkl')

#%%
# =============================================================================
# Scatter - AutoBCI vs. (MCUB, MCSB)
# =============================================================================
acc_bu = BU['acc']
acc_sb = SB['acc']
acc_as = r['media']
plt.figure(figsize=(20,6), facecolor='mintcream')
plt.grid(axis='x', **dict(ls='--', alpha=0.3))
# plt.title('Gráfico de dispersão do desempenho da classificação individual')

plt.subplot(121)
plt.scatter(np.asarray(acc_bu).reshape(-1,1), np.asarray(acc_as).reshape(-1,1), facecolors = 'c', marker = 'o', s=50, alpha=.9, edgecolors='firebrick', zorder=3)
plt.scatter(round(acc_bu.mean(),2), round(acc_as.mean(),2), facecolors = 'dodgerblue', marker = 'o', s=100, alpha=1, edgecolors='darkblue', label='Average accuracy', zorder=5)
plt.plot(np.linspace(30, 110, 1000), np.linspace(30, 110, 1000), color='dimgray', linewidth=1, linestyle='--', zorder=0)
plt.plot(np.ones(1000)*round(acc_bu.mean(),2), np.linspace(30, round(acc_as.mean(),2), 1000), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 
plt.plot(np.linspace(30, round(acc_bu.mean(),2), 1000), np.ones(1000)*round(acc_as.mean(),2), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 
plt.ylim((38, 102))
plt.xlim((38, 102))
plt.xticks(np.arange(40, 102, 5)) 
plt.yticks(np.arange(40, 102, 5))
plt.legend(loc='lower right', fontsize=12)
plt.ylabel(r'$Auto$BCI accuracy', fontsize=14)
plt.xlabel('MCUB accuracy', fontsize=14)

plt.subplot(122)
plt.scatter(np.asarray(acc_sb).reshape(-1,1), np.asarray(acc_as).reshape(-1,1), facecolors = 'c', marker = 'o', s=50, alpha=.9, edgecolors='firebrick', zorder=3)
plt.scatter(round(acc_sb.mean(),2), round(acc_as.mean(),2), facecolors = 'dodgerblue', marker = 'o', s=100, alpha=1, edgecolors='darkblue', label='Average accuracy', zorder=5)
plt.plot(np.linspace(30, 110, 1000), np.linspace(30, 110, 1000), color='dimgray', linewidth=1, linestyle='--', zorder=0)
plt.plot(np.ones(1000)*round(acc_sb.mean(),2), np.linspace(30, round(acc_as.mean(),2), 1000), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 
plt.plot(np.linspace(30, round(acc_sb.mean(),2), 1000), np.ones(1000)*round(acc_as.mean(),2), color='dimgray', linewidth=.7, linestyle=':', zorder=0) 
plt.ylim((38, 102))
plt.xlim((38, 102))
plt.xticks(np.arange(40, 102, 5)) 
plt.yticks(np.arange(40, 102, 5))
plt.legend(loc='lower right', fontsize=12)
plt.ylabel(r'$Auto$BCI accuracy', fontsize=14)
plt.xlabel('MCSB accuracy', fontsize=14)

plt.tight_layout(w_pad=3, h_pad=0) # pad=0.4, , 

# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_scatter_compare.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%
# =============================================================================
# Boxplot - AutoBCI vs. (MCUB, MCSB) - Subject
# =============================================================================
plt.figure(figsize=(20,6), facecolor='mintcream')

plt.subplot(131)
df = pd.DataFrame()
for suj in range(1, 10):
    df.insert(df.shape[-1], 'S{}'.format(suj), np.asarray(r[r.suj == suj].media))
plt.grid(axis='y', **dict(ls='--', alpha=0.3))
plt.plot(np.linspace(0.5, 9.5, 100), np.ones(100)*r.media.mean(), 
         color='r', linewidth=2.5, alpha=.8, linestyle=':', label=r'$Auto$BCI average accuracy', zorder=0)
plt.boxplot(df, vert=True, showfliers=True, patch_artist=True, showcaps=True, zorder=1,
            medianprops=dict(lw=1.5, color='navy'), whiskerprops=dict(color='darkslateblue', ls='-', lw=1, alpha=1),
            boxprops=dict(color='darkslateblue', facecolor='paleturquoise', lw=1, alpha=1, hatch=''))
plt.xticks(range(1,10), [ str(suj) for suj in range(1,10) ], fontsize=11)
plt.yticks(np.arange(50, 102, 5), fontsize=11)
plt.xlabel('Subject', size=14); plt.ylabel(r'Auto$BCI$ accuracy', size=14)
plt.xlim((0.5,9.5)); plt.ylim((48, 102))
plt.legend(loc='lower right', fontsize=13, ncol=1, borderaxespad=0.2, framealpha=0.99, labelspacing=0.2, prop=dict(size='13')) # bbox_to_anchor=(-0.1,1.1)

plt.subplot(132)
df = pd.DataFrame()
for suj in range(1, 10):
    df.insert(df.shape[-1], 'S{}'.format(suj), np.asarray(BU[BU.suj == suj].acc))
plt.grid(axis='y', **dict(ls='--', alpha=0.3))
plt.plot(np.linspace(0.5, 9.5, 100), np.ones(100)*BU.acc.mean(), 
         color='k', linewidth=2.5, alpha=.8, linestyle=':', label=r'MCUB average accuracy', zorder=0)
plt.boxplot(df, vert=True, showfliers=True, patch_artist=True, showcaps=True, zorder=1,
            medianprops=dict(lw=1.5, color='white'), whiskerprops=dict(color='darkslateblue', ls='-', lw=2, alpha=1),
            boxprops=dict(color='darkslateblue', facecolor='darksalmon', lw=1.5, alpha=1, hatch=''))
plt.xticks(range(1,10), [ str(suj) for suj in range(1,10) ], fontsize=11)
plt.yticks(np.arange(50, 102, 5), fontsize=11)
plt.xlabel('Subject', size=14); plt.ylabel(r'MCUB accuracy', size=14)
plt.xlim((0.5,9.5)); plt.ylim((48, 102))
plt.legend(loc='lower right', fontsize=13, ncol=1, borderaxespad=0.2, framealpha=0.99, labelspacing=0.2, prop=dict(size='13')) # bbox_to_anchor=(-0.1,1.1)

plt.subplot(133)
df = pd.DataFrame()
for suj in range(1, 10):
    df.insert(df.shape[-1], 'S{}'.format(suj), np.asarray(SB[SB.suj == suj].acc))
plt.grid(axis='y', **dict(ls='--', alpha=0.3))
plt.plot(np.linspace(0.5, 9.5, 100), np.ones(100)*SB.acc.mean(), 
         color='forestgreen', linewidth=2.5, alpha=.8, linestyle=':', label=r'MCSB average accuracy', zorder=0)
plt.boxplot(df, vert=True, showfliers=True, patch_artist=True, showcaps=True, zorder=1,
            medianprops=dict(lw=1.5, color='navy'), whiskerprops=dict(color='darkslateblue', ls='-', lw=2, alpha=1),
            boxprops=dict(color='darkslateblue', facecolor='khaki', lw=1.5, alpha=1, hatch=''))
plt.xticks(range(1,10), [ str(suj) for suj in range(1,10) ], fontsize=11)
plt.yticks(np.arange(50, 102, 5), fontsize=11)
plt.xlabel('Subject', size=14); plt.ylabel(r'MCSB accuracy', size=14)
plt.xlim((0.5,9.5)); plt.ylim((48, 102))
plt.legend(loc='lower right', fontsize=13, ncol=1, borderaxespad=0.2, framealpha=0.99, labelspacing=0.2, prop=dict(size='13')) # bbox_to_anchor=(-0.1,1.1)

plt.tight_layout(w_pad=3, h_pad=0) # pad=0.4, ,

# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_boxplot_compare_subjects.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Boxplot - AutoBCI vs. (MCUB, MCSB) - Classes
# =============================================================================
plt.figure(figsize=(20,6), facecolor='mintcream')

plt.subplot(131)
df = pd.DataFrame()
for c,l in zip(['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'],['LHxRH', 'LHxFT', 'LHxTG', 'RHxFT', 'RHxTG', 'FTxTG']):
    df.insert(df.shape[-1], '{}'.format(l), np.asarray(r[r.classes == c].media))
plt.grid(axis='y', **dict(ls='--', alpha=0.3))
plt.plot(np.linspace(0.5, 6.5, 100), np.ones(100)*r.media.mean(), 
         color='r', linewidth=2.5, alpha=.8, linestyle=':', label=r'$Auto$BCI average accuracy', zorder=0)
plt.boxplot(df, vert=True, showfliers=True, patch_artist=True, showcaps=True, zorder=1,
            medianprops=dict(lw=1.5, color='navy'), whiskerprops=dict(color='darkslateblue', ls='-', lw=1, alpha=1),
            boxprops=dict(color='darkslateblue', facecolor='paleturquoise', lw=1, alpha=1, hatch=''))
plt.xticks(range(1,7), ['LH vs. RH', 'LH vs. FT', 'LH vs. TG', 'RH vs. FT', 'RH vs. TG', 'FT vs. TG'], fontsize=11)
plt.yticks(np.arange(50, 102, 5), fontsize=11)
plt.xlabel('MI class pairs', size=14); plt.ylabel(r'Auto$BCI$ accuracy', size=14)
plt.xlim((0.5,6.5)); plt.ylim((48, 102))
plt.legend(loc='lower right', fontsize=13, ncol=1, borderaxespad=0.2, framealpha=0.99, labelspacing=0.2, prop=dict(size='13')) # bbox_to_anchor=(-0.1,1.1),

plt.subplot(132)
df = pd.DataFrame()
for c,l in zip(['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'],['LHxRH', 'LHxFT', 'LHxTG', 'RHxFT', 'RHxTG', 'FTxTG']):
    df.insert(df.shape[-1], '{}'.format(l), np.asarray(BU[BU.classes == c].acc))
plt.grid(axis='y', **dict(ls='--', alpha=0.3))
plt.plot(np.linspace(0.5, 6.5, 100), np.ones(100)*BU.acc.mean(), 
         color='k', linewidth=2.5, alpha=.8, linestyle=':', label=r'MCUB average accuracy', zorder=0)
plt.boxplot(df, vert=True, showfliers=True, patch_artist=True, showcaps=True, zorder=1,
            medianprops=dict(lw=1.5, color='white'), whiskerprops=dict(color='darkslateblue', ls='-', lw=2, alpha=1),
            boxprops=dict(color='darkslateblue', facecolor='darksalmon', lw=1.5, alpha=1, hatch=''))
plt.xticks(range(1,7), ['LH vs. RH', 'LH vs. FT', 'LH vs. TG', 'RH vs. FT', 'RH vs. TG', 'FT vs. TG'], fontsize=11)
plt.yticks(np.arange(50, 102, 5), fontsize=11)
plt.xlabel('MI class pairs', size=14); plt.ylabel(r'MCUB accuracy', size=14)
plt.xlim((0.5,6.5)); plt.ylim((48, 102))
plt.legend(loc='lower right', fontsize=13, ncol=1, borderaxespad=0.2, framealpha=0.99, labelspacing=0.2, prop=dict(size='13')) # bbox_to_anchor=(-0.1,1.1)

plt.subplot(133)
df = pd.DataFrame()
for c,l in zip(['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'],['LHxRH', 'LHxFT', 'LHxTG', 'RHxFT', 'RHxTG', 'FTxTG']):
    df.insert(df.shape[-1], '{}'.format(l), np.asarray(SB[SB.classes == c].acc))
plt.grid(axis='y', **dict(ls='--', alpha=0.3))
plt.plot(np.linspace(0.5, 6.5, 100), np.ones(100)*SB.acc.mean(), 
         color='forestgreen', linewidth=2.5, alpha=.8, linestyle=':', label=r'MCSB average accuracy', zorder=0)
plt.boxplot(df, vert=True, showfliers=True, patch_artist=True, showcaps=True, zorder=1,
            medianprops=dict(lw=1.5, color='navy'), whiskerprops=dict(color='darkslateblue', ls='-', lw=2, alpha=1),
            boxprops=dict(color='darkslateblue', facecolor='khaki', lw=1.5, alpha=1, hatch=''))
plt.xticks(range(1,7), ['LH vs. RH', 'LH vs. FT', 'LH vs. TG', 'RH vs. FT', 'RH vs. TG', 'FT vs. TG'], fontsize=11)
plt.yticks(np.arange(50, 102, 5), fontsize=11)
plt.xlabel('MI class pairs', size=14); plt.ylabel(r'MCSB accuracy', size=14)
plt.xlim((0.5,6.5)); plt.ylim((48, 102))
plt.legend(loc='lower right', fontsize=13, ncol=1, borderaxespad=0.2, framealpha=0.99, labelspacing=0.2, prop=dict(size='13')) # bbox_to_anchor=(-0.1,1.1)

plt.tight_layout(w_pad=3, h_pad=0) # pad=0.4, ,

# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_boxplot_compare_classes.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Boxplot Accuracy AutoBCI vs. (MCUB, MCSB) - Subplots per Subjects
# =============================================================================
plt.figure(figsize=(15,10), facecolor='mintcream')

for suj in range(1, 10):
    
    
    plt.subplot(3,3,suj)
    plt.grid(axis='y', **dict(ls='--', alpha=0.3))

    x = [ r[r.suj == suj][r.classes == c].media.mean() for c in ['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'] ]
    
    df = pd.DataFrame()
    df.insert(df.shape[-1], 'AutoBCI', np.asarray(x))
    df.insert(df.shape[-1], 'MCUB', np.asarray(BU[BU.suj == suj].acc))
    df.insert(df.shape[-1], 'MCSB', np.asarray(SB[SB.suj == suj].acc))

    
    plt.boxplot(df, vert=True, showfliers=True, patch_artist=True, showcaps=True, zorder=1,
                medianprops=dict(lw=1.5, color='navy'), whiskerprops=dict(color='darkslateblue', ls='-', lw=1, alpha=1),
                boxprops=dict(color='darkslateblue', facecolor='aliceblue', lw=1, alpha=1, hatch=''))
       
    plt.xticks(np.arange(1,4), ['AutoBCI', 'MCUB', 'MCSB'], fontsize=11)
    plt.yticks(np.arange(50, 102, 5), fontsize=11)
    plt.xlim((0,4)); plt.ylim((48,102))
    
    plt.title(f"Subject {suj}", fontsize=18)
    plt.tight_layout(w_pad=2, h_pad=3.0) # pad=0.4, , 
    
    plt.xlabel('Configuration approach', size=13)
    plt.ylabel('Accuracy', size=13)
    
    plt.tight_layout(w_pad=2, h_pad=3) # pad=0.4, ,
    
# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_acc_subplots_compare_subjects.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Boxplot Accuracy AutoBCI vs. (MCUB, MCSB) - Subplots per Classes
# =============================================================================
plt.figure(figsize=(15,8), facecolor='mintcream')


for i,c,l in zip(range(1,7),['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'],['LHxRH', 'LHxFT', 'LHxTG', 'RHxFT', 'RHxTG', 'FTxTG']):
    
    plt.subplot(2,3,i)
    plt.grid(axis='y', **dict(ls='--', alpha=0.3))

    x = [ r[r.suj == suj][r.classes == c].media.mean() for suj in range(1, 10) ]
    
    df = pd.DataFrame()
    df.insert(df.shape[-1], 'AutoBCI', np.asarray(x))
    df.insert(df.shape[-1], 'MCUB', np.asarray(BU[BU.classes == c].acc))
    df.insert(df.shape[-1], 'MCSB', np.asarray(SB[SB.classes == c].acc))
    
    plt.boxplot(df, vert=True, showfliers=True, patch_artist=True, showcaps=True, zorder=1,
                medianprops=dict(lw=1.5, color='navy'), whiskerprops=dict(color='darkslateblue', ls='-', lw=1, alpha=1),
                boxprops=dict(color='darkslateblue', facecolor='aliceblue', lw=1, alpha=1, hatch=''))
       
    plt.xticks(np.arange(1,4), ['AutoBCI', 'MCUB', 'MCSB'], fontsize=11)
    plt.yticks(np.arange(50, 102, 5), fontsize=11)
    plt.xlim((0,4)); plt.ylim((48,102))
    
    plt.title(f"{l.split('x')[0]} vs. {l.split('x')[1]}", fontsize=18)
    plt.tight_layout(w_pad=2, h_pad=3.0) # pad=0.4, , 
    
    plt.xlabel('Configuration approach', size=13)
    plt.ylabel('Accuracy', size=13)
    
    plt.tight_layout(w_pad=2, h_pad=3) # pad=0.4, ,
    
# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_acc_subplots_compare_classes.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%
# =============================================================================
# Proporção classificadores Pizza
# =============================================================================
cores = ['steelblue', 'turquoise', 'lightsalmon', 'darkgray', 'mediumorchid']

labels = np.unique(S.clf, return_counts=True)[0]
sizes = np.unique(S.clf, return_counts=True)[1]
explode = (0.02, 0.02, 0.02, 0.02, 0.02)  # only "explode" the 2nd slice

def func(pct, allvals):
    print(pct)
    absolute = pct/100.*np.sum(allvals)
    return "{:.1f}%\n({:d})".format(pct, int(round(absolute,0)))

fig1, ax1 = plt.subplots(figsize=(10,6), subplot_kw=dict(aspect="equal"))
wedges, texts, autotexts = ax1.pie(sizes, explode=explode, colors = cores, labels=labels, autopct=lambda pct: func(pct, sizes), 
                                    shadow=False, startangle=90, textprops=dict(fontsize=10, color="k")) # '%1.1f%%'
ax1.legend(wedges, labels, title=r"${\Phi}$ space", loc="center left", bbox_to_anchor=(.8, 0, 0.5, .25), fontsize=12)
plt.setp(autotexts, size=12, weight="bold") # bold
plt.setp(texts, size=12, weight="bold")
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_classifier_pie.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Histogramas classificadores
# =============================================================================
plt.rcParams["font.family"] = "cursive"
 
plt.figure(figsize=(20,5), facecolor='mintcream')

plt.subplot(1,2,1)
plt.grid(True, axis='y', **dict(ls='--', alpha=0.6), zorder=0)
for i,c,l in zip(range(0,6),['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'],['LHxRH', 'LHxFT', 'LHxTG', 'RHxFT', 'RHxTG', 'FTxTG']):
    p = np.unique(S[S.classes == c].clf, return_counts=True)[1]/len(S[S.classes == c].clf)*100
    plt.bar(l, np.sum(p[:1]), width=.7, zorder=5, color=cores[0], label='')
    plt.bar(l, np.sum(p[:2]), width=.7, zorder=4, color=cores[1], label='')
    plt.bar(l, np.sum(p[:3]), width=.7, zorder=3, color=cores[2], label='')
    plt.bar(l, np.sum(p[:4]), width=.7, zorder=2, color=cores[3], label='')
    plt.bar(l, np.sum(p[:5]), width=.7, zorder=1, color=cores[4], label='')

plt.xlim((-0.5, 5.5)); plt.ylim((0, 102))
plt.xlabel(r'MI class pairs', fontsize=14)
plt.ylabel('Occurrence (%)', fontsize=14)
plt.yticks(np.arange(0, 110, 10), fontsize=12)
plt.xticks(range(0,6), ['LH vs. RH', 'LH vs. FT', 'LH vs. TG', 'RH vs. FT', 'RH vs. TG', 'FT vs. TG'], fontsize=12)

plt.subplot(1,2,2)
plt.grid(True, axis='y', **dict(ls='--', alpha=0.6), zorder=0)
for suj in range(1,10):
    p = np.unique(S[S.suj == suj].clf, return_counts=True)[1]/len(S[S.suj == suj].clf)*100
    plt.bar(suj, np.sum(p[:1]), width=.7, zorder=5, color=cores[0], label='')
    plt.bar(suj, np.sum(p[:2]), width=.7, zorder=4, color=cores[1], label='')
    plt.bar(suj, np.sum(p[:3]), width=.7, zorder=3, color=cores[2], label='')
    plt.bar(suj, np.sum(p[:4]), width=.7, zorder=2, color=cores[3], label='')
    plt.bar(suj, np.sum(p[:5]), width=.7, zorder=1, color=cores[4], label='')

plt.xlim((0.5, 9.5)); plt.ylim((0, 102))
plt.xlabel(r'Subjects', fontsize=14)
plt.ylabel('Occurrence (%)', fontsize=14)
plt.yticks(np.arange(0, 110, 10), fontsize=12)
plt.xticks(range(1,10), fontsize=12)
# plt.legend(loc='best', fontsize=12)

plt.tight_layout(w_pad=3, h_pad=0) # pad=0.4, ,

# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_classifier_hist.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%
# =============================================================================
# Boxplot Janela
# =============================================================================
plt.figure(figsize=(10,6), facecolor='mintcream')
plt.grid(axis='x', **dict(ls='--', alpha=0.3))
plt.xlim((-0.2,4.2))
plt.ylim((0.5,9.5))
plt.yticks(range(1,10), [ str(suj) for suj in range(1,10) ], fontsize=11)
plt.xticks(np.arange(0,4.5,0.5), ['Cue', 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], fontsize=11)
plt.ylabel('Subject', size=14)
plt.xlabel('Time (sec)', size=14)

df = pd.DataFrame()
for suj in range(1, 10):
    # x = np.asarray(S[S.suj == suj][R1.classes == '1 2'].tl.append(S[S.suj == suj][R1.classes == '1 2'].tu, ignore_index=True))
    x = np.asarray(S[S.suj == suj].tl.append(S[S.suj == suj].tu, ignore_index=True))
    df.insert(df.shape[-1], 'S{}'.format(suj), x)

plt.fill_between(np.linspace(0.5, 2.5, 100), np.zeros(100), np.ones(100)*9.5, color='lemonchiffon', lw=3, alpha=.8, zorder=0)
plt.plot(np.zeros(100), np.linspace(0, 9.5, 100), color='crimson', linewidth=1.3, alpha=.8, linestyle='--', label=r'Dica', zorder=2)

plt.boxplot(df, vert=False, showfliers=False, patch_artist=True, showcaps=False, zorder=1,
            medianprops=dict(lw=0), whiskerprops=dict(color='darkslateblue', ls='-', lw=2, alpha=1),
            boxprops=dict(color='darkslateblue', facecolor='aliceblue', lw=1.5, alpha=1, hatch='//////'))
# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_win_box_subjects.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Boxplot Banda
# =============================================================================
plt.figure(figsize=(10,6), facecolor='mintcream')
plt.grid(axis='x', **dict(ls='--', alpha=0.3))

df = pd.DataFrame()
for suj in range(1, 10):
    # x = np.asarray(S[S.suj == suj][R1.classes == '1 2'].fl.append(S[S.suj == suj][R1.classes == '1 2'].fu, ignore_index=True))
    x = np.asarray(S[S.suj == suj].fl.append(S[S.suj == suj].fu, ignore_index=True))
    df.insert(df.shape[-1], 'S{}'.format(suj), x)

plt.fill_between(np.linspace(4, 40, 100), np.zeros(100), np.ones(100)*9.5, color='lemonchiffon', lw=3, alpha=.6, zorder=0)
plt.fill_between(np.linspace(8, 30, 100), np.zeros(100), np.ones(100)*9.5, color='powderblue', lw=3, alpha=.7, zorder=1)
plt.boxplot(df, vert=False, showfliers=False, patch_artist=True, showcaps=False, zorder=2,
            medianprops=dict(lw=0), whiskerprops=dict(color='firebrick', ls='-', lw=2, alpha=1),
            boxprops=dict(color='firebrick', facecolor='tomato', lw=1.5, alpha=1, hatch='//////'))

plt.yticks(range(1,10), [ str(suj) for suj in range(1,10) ], fontsize=11)
plt.xticks(np.arange(0,51,2), fontsize=11)
plt.xlabel('Frequency (Hz)', size=14); plt.ylabel('Subject', size=14)
plt.xlim((-2,52)); plt.ylim((0.5,9.5))

# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_freq_box_subjects.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Boxplot Janela + Banda per subjects
# =============================================================================
plt.figure(figsize=(20,5), facecolor='mintcream')

plt.subplot(121)
df = pd.DataFrame()
for suj in range(1, 10):
    # x = np.asarray(S[S.suj == suj][R1.classes == '1 2'].tl.append(S[S.suj == suj][R1.classes == '1 2'].tu, ignore_index=True))
    x = np.asarray(S[S.suj == suj].tl.append(S[S.suj == suj].tu, ignore_index=True))
    df.insert(df.shape[-1], 'S{}'.format(suj), x)
plt.grid(axis='x', **dict(ls='--', alpha=0.3))
plt.fill_between(np.linspace(0.5, 2.5, 100), np.zeros(100), np.ones(100)*9.5, color='lemonchiffon', lw=3, alpha=.8, zorder=0)
plt.plot(np.zeros(100), np.linspace(0, 9.5, 100), color='crimson', linewidth=1.3, alpha=.8, linestyle='--', label=r'Dica', zorder=2)
plt.boxplot(df, vert=False, showfliers=False, patch_artist=True, showcaps=False, zorder=1,
            medianprops=dict(lw=0), whiskerprops=dict(color='darkslateblue', ls='-', lw=2, alpha=1),
            boxprops=dict(color='darkslateblue', facecolor='aliceblue', lw=1.5, alpha=1, hatch='//////'))
plt.yticks(range(1,10), [ str(suj) for suj in range(1,10) ], fontsize=11)
plt.xticks(np.arange(0,4.5,0.5), ['Cue', 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], fontsize=11)
plt.xlabel('Time (sec)', size=14); plt.ylabel('Subject', size=14)
plt.xlim((-0.2,4.2)); plt.ylim((0.5,9.5))

plt.subplot(122)
df = pd.DataFrame()
for suj in range(1, 10):
    # x = np.asarray(S[S.suj == suj][R1.classes == '1 2'].fl.append(S[S.suj == suj][R1.classes == '1 2'].fu, ignore_index=True))
    x = np.asarray(S[S.suj == suj].fl.append(S[S.suj == suj].fu, ignore_index=True))
    df.insert(df.shape[-1], 'S{}'.format(suj), x)
plt.grid(axis='x', **dict(ls='--', alpha=0.3))
plt.fill_between(np.linspace(4, 40, 100), np.zeros(100), np.ones(100)*9.5, color='lemonchiffon', lw=3, alpha=.6, zorder=0)
plt.fill_between(np.linspace(8, 30, 100), np.zeros(100), np.ones(100)*9.5, color='powderblue', lw=3, alpha=.7, zorder=1)
plt.boxplot(df, vert=False, showfliers=False, patch_artist=True, showcaps=False, zorder=2,
            medianprops=dict(lw=0), whiskerprops=dict(color='firebrick', ls='-', lw=2, alpha=1),
            boxprops=dict(color='firebrick', facecolor='tomato', lw=1.5, alpha=1, hatch='//////'))
plt.yticks(range(1,10), [ str(suj) for suj in range(1,10) ], fontsize=11)
plt.xticks(np.arange(0,51,2), fontsize=11)
plt.xlabel('Frequency (Hz)', size=14); plt.ylabel('Subject', size=14)
plt.xlim((-2,52)); plt.ylim((0.5,9.5))

plt.tight_layout(w_pad=3, h_pad=0) # pad=0.4, , 

# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_win_freq_box_subjects.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%
# =============================================================================
# Boxplot Janela + Banda per classes
# =============================================================================
plt.figure(figsize=(20,5), facecolor='mintcream')

plt.subplot(121)
df = pd.DataFrame()
for i,c,l in zip(range(0,6),['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'],['LHxRH', 'LHxFT', 'LHxTG', 'RHxFT', 'RHxTG', 'FTxTG']):
    x = np.asarray(S[S.classes == c].tl.append(S[S.classes == c].tu, ignore_index=True))
    df.insert(df.shape[-1], '{}'.format(l), x)
plt.grid(axis='x', **dict(ls='--', alpha=0.3))
plt.fill_between(np.linspace(0.5, 2.5, 100), np.zeros(100), np.ones(100)*6.5, color='lemonchiffon', lw=3, alpha=.8, zorder=0)
plt.plot(np.zeros(100), np.linspace(0, 6.5, 100), color='crimson', linewidth=1.3, alpha=.8, linestyle='--', label=r'Dica', zorder=2)
plt.boxplot(df, vert=False, showfliers=False, patch_artist=True, showcaps=False, zorder=1,
            medianprops=dict(lw=0), whiskerprops=dict(color='darkslateblue', ls='-', lw=2, alpha=1),
            boxprops=dict(color='darkslateblue', facecolor='aliceblue', lw=1.5, alpha=1, hatch='//////'))
plt.yticks(range(1,7), ['LH vs. RH', 'LH vs. FT', 'LH vs. TG', 'RH vs. FT', 'RH vs. TG', 'FT vs. TG'], fontsize=11)
plt.xticks(np.arange(0,4.5,0.5), ['Cue', 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], fontsize=11)
plt.xlabel('Time (sec)', size=14); plt.ylabel('MI class pairs', size=14)
plt.xlim((-0.2,4.2)); plt.ylim((0.5,6.5))

plt.subplot(122)
df = pd.DataFrame()
for i,c,l in zip(range(0,6),['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'],['LHxRH', 'LHxFT', 'LHxTG', 'RHxFT', 'RHxTG', 'FTxTG']):
    # x = np.asarray(S[S.suj == suj][R1.classes == '1 2'].fl.append(S[S.suj == suj][R1.classes == '1 2'].fu, ignore_index=True))
    x = np.asarray(S[S.classes == c].fl.append(S[S.classes == c].fu, ignore_index=True))
    df.insert(df.shape[-1], '{}'.format(l), x)
plt.grid(axis='x', **dict(ls='--', alpha=0.3))
plt.fill_between(np.linspace(4, 40, 100), np.zeros(100), np.ones(100)*6.5, color='lemonchiffon', lw=3, alpha=.6, zorder=0)
plt.fill_between(np.linspace(8, 30, 100), np.zeros(100), np.ones(100)*6.5, color='powderblue', lw=3, alpha=.7, zorder=1)
plt.boxplot(df, vert=False, showfliers=False, patch_artist=True, showcaps=False, zorder=2,
            medianprops=dict(lw=0), whiskerprops=dict(color='firebrick', ls='-', lw=2, alpha=1),
            boxprops=dict(color='firebrick', facecolor='tomato', lw=1.5, alpha=1, hatch='//////'))
plt.yticks(range(1,7), ['LH vs. RH', 'LH vs. FT', 'LH vs. TG', 'RH vs. FT', 'RH vs. TG', 'FT vs. TG'], fontsize=11)
plt.xticks(np.arange(0,51,2), fontsize=11)
plt.xlabel('Frequency (Hz)', size=14); plt.ylabel('MI class pairs', size=14)
plt.xlim((-2,52)); plt.ylim((0.5,6.5))

plt.tight_layout(w_pad=3, h_pad=0) # pad=0.4, , 

# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_win_freq_box_classes.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


#%%
# =============================================================================
# Boxplot Janela - Subplots per subjects
# =============================================================================
plt.figure(figsize=(20,10), facecolor='mintcream')

for suj in range(1, 10):
    plt.subplot(3,3,suj)
    plt.grid(axis='x', **dict(ls='--', alpha=0.3))
        
    df = pd.DataFrame()
    for c,l in zip(['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'],['LHxRH', 'LHxFT', 'LHxTG', 'RHxFT', 'RHxTG', 'FTxTG']):
        x = np.asarray(S[S.suj == suj][R1.classes == c].tl.append(
            S[S.suj == suj][R1.classes == c].tu, ignore_index=True))
        df.insert(df.shape[-1], '{}'.format(l), x)
    
    plt.fill_between(np.linspace(0.5, 2.5, 100), np.zeros(100), np.ones(100)*6.5, color='lemonchiffon', lw=3, alpha=.8, zorder=0)
    plt.plot(np.zeros(100), np.linspace(0, 6.5, 100), color='crimson', linewidth=1.3, alpha=.8, linestyle='--', label=r'Dica', zorder=2)

    plt.boxplot(df, vert=False, showfliers=False, patch_artist=True, showcaps=False, zorder=1,
                medianprops=dict(lw=0), whiskerprops=dict(color='darkslateblue', ls='-', lw=2, alpha=1), 
                boxprops=dict(color='darkslateblue', facecolor='aliceblue', lw=1.5, alpha=1, hatch='//////'))
       
    plt.xticks(np.arange(0,4.5,0.5), ['Cue', 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], fontsize=11)
    plt.yticks(range(1,7), ['LH vs. RH', 'LH vs. FT', 'LH vs. TG', 'RH vs. FT', 'RH vs. TG', 'FT vs. TG'], fontsize=11)
    plt.xlim((-0.2,4.2)); plt.ylim((0.5,6.5))
    
    plt.title(f"Subject {suj}", fontsize=18)
    plt.tight_layout(w_pad=2, h_pad=3.0) # pad=0.4, , 

    plt.xlabel('Time (sec)', size=13)
    plt.ylabel('MI class pairs', size=13)
     
# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_win_subbox_subjects.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Boxplot Janela - Subplots per classes
# =============================================================================
fig = plt.figure(figsize=(18,8), facecolor='mintcream')

for i,c,l in zip(range(1,7),['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'],['LHxRH', 'LHxFT', 'LHxTG', 'RHxFT', 'RHxTG', 'FTxTG']):
    plt.subplot(2,3,i)
    plt.grid(axis='x', **dict(ls='--', alpha=0.3))
        
    df = pd.DataFrame()
    for suj in range(1, 10):
        x = np.asarray(S[S.suj == suj][R1.classes == c].tl.append(
            S[S.suj == suj][R1.classes == c].tu, ignore_index=True))
        df.insert(df.shape[-1], '{}'.format(suj), x)
    
    plt.fill_between(np.linspace(0.5, 2.5, 100), np.zeros(100), np.ones(100)*9.5, color='lemonchiffon', lw=3, alpha=.8, zorder=0)
    plt.plot(np.zeros(100), np.linspace(0, 9.5, 100), color='crimson', linewidth=1.3, alpha=.8, linestyle='--', label=r'Dica', zorder=2)
    plt.boxplot(df, vert=False, showfliers=False, patch_artist=True, showcaps=False, zorder=1,
                medianprops=dict(lw=0), whiskerprops=dict(color='darkslateblue', ls='-', lw=2, alpha=1), 
                boxprops=dict(color='darkslateblue', facecolor='aliceblue', lw=1.5, alpha=1, hatch='//////'))
    
    plt.xticks(np.arange(0,4.5,0.5), ['Cue', 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], fontsize=11)
    plt.yticks(range(1,10), fontsize=11)
    plt.xlim((-0.2,4.2)); plt.ylim((0.5,9.5))
    
    plt.title(f"{l.split('x')[0]} vs. {l.split('x')[1]}", fontsize=18)
    plt.tight_layout(w_pad=2, h_pad=3.0) # pad=0.4, , 

    plt.xlabel('Time (sec)', size=13)
    plt.ylabel('Subject', size=13)
    
# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_win_subbox_classes.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Boxplot Banda - Subplots per subjects
# =============================================================================
plt.figure(figsize=(20,10), facecolor='mintcream')

for suj in range(1, 10):
    plt.subplot(3,3,suj)
    plt.grid(axis='x', **dict(ls='--', alpha=0.3))
        
    df = pd.DataFrame()
    for c,l in zip(['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'],['LHxRH', 'LHxFT', 'LHxTG', 'RHxFT', 'RHxTG', 'FTxTG']):
        x = np.asarray(S[S.suj == suj][R1.classes == c].fl.append(
            S[S.suj == suj][R1.classes == c].fu, ignore_index=True))
        df.insert(df.shape[-1], '{}'.format(l), x)
    
    plt.fill_between(np.linspace(4, 40, 100), np.zeros(100), np.ones(100)*6.5, color='lemonchiffon', lw=3, alpha=.6, zorder=0)
    plt.fill_between(np.linspace(8, 30, 100), np.zeros(100), np.ones(100)*6.5, color='powderblue', lw=3, alpha=.7, zorder=1)
    plt.boxplot(df, vert=False, showfliers=False, patch_artist=True, showcaps=False, zorder=2,
                medianprops=dict(lw=0), whiskerprops=dict(color='firebrick', ls='-', lw=2, alpha=1), 
                boxprops=dict(color='firebrick', facecolor='tomato', lw=1.5, alpha=1, hatch='//////'))
    
    plt.xticks(np.arange(0,51,2), fontsize=11, rotation=60)
    plt.yticks(range(1,7), ['LH vs. RH', 'LH vs. FT', 'LH vs. TG', 'RH vs. FT', 'RH vs. TG', 'FT vs. TG'], fontsize=11)
    plt.xlim((-2,52)); plt.ylim((0.5,6.5))
    
    plt.title(f"Subject {suj}", fontsize=18)
    plt.tight_layout(w_pad=2, h_pad=3.0) # pad=0.4, , 

    plt.xlabel('Frequency (Hz)', size=13)
    plt.ylabel('MI class pairs', size=13)
    
# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_freq_subbox_subjects.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
# =============================================================================
# Boxplot Banda - Subplots per classes
# =============================================================================
plt.figure(figsize=(18,8), facecolor='mintcream')

for i,c,l in zip(range(1,7),['1 2', '1 3', '1 4', '2 3', '2 4', '3 4'],['LHxRH', 'LHxFT', 'LHxTG', 'RHxFT', 'RHxTG', 'FTxTG']):
    plt.subplot(2,3,i)
    plt.grid(axis='x', **dict(ls='--', alpha=0.3))
        
    df = pd.DataFrame()
    for suj in range(1, 10):
        x = np.asarray(S[S.suj == suj][R1.classes == c].fl.append(
            S[S.suj == suj][R1.classes == c].fu, ignore_index=True))
        df.insert(df.shape[-1], '{}'.format(suj), x)
    
    plt.fill_between(np.linspace(4, 40, 100), np.zeros(100), np.ones(100)*9.5, color='lemonchiffon', lw=3, alpha=.6, zorder=0)
    plt.fill_between(np.linspace(8, 30, 100), np.zeros(100), np.ones(100)*9.5, color='powderblue', lw=3, alpha=.7, zorder=1)
    plt.boxplot(df, vert=False, showfliers=False, patch_artist=True, showcaps=False, zorder=2,
                medianprops=dict(lw=0), whiskerprops=dict(color='firebrick', ls='-', lw=2, alpha=1), 
                boxprops=dict(color='firebrick', facecolor='tomato', lw=1.5, alpha=1, hatch='//////'))
    
    plt.xticks(np.arange(0,51,2), fontsize=11, rotation=60)
    plt.yticks(range(1,10), fontsize=11)
    plt.xlim((-2,52)); plt.ylim((0.5,9.5))
    
    plt.title(f"{l.split('x')[0]} vs. {l.split('x')[1]}", fontsize=18)
    plt.tight_layout(w_pad=2, h_pad=3.0) # pad=0.4, , 

    plt.xlabel('Frequency (Hz)', size=13)
    plt.ylabel('Subject', size=13)
    
# plt.savefig('G:/Meu Drive/devs/BCI/results/trials/r_freq_subbox_classes.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

#%%
plt.figure(figsize=(20,13), facecolor='mintcream')
plt.subplot(321)
plt.subplot(322)
plt.subplot(323)
plt.subplot(324)
plt.subplot(325)
plt.subplot(326)