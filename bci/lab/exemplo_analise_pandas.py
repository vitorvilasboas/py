# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

ds = 'IV2a'
S = pd.read_pickle('/home/vboas/cloud/auto_setup_results/' + ds + '/setup_list_final.pickle')

print(S['acc'].mean(), S['acc'].median(), S['acc'].std(), S['acc'].max(), S['acc'].min())







suj = np.ravel(range(1,10))
ncomp = np.ravel(range(2, 23, 2))
sbands = np.ravel(range(1, 23))

R = pd.read_pickle(os.path.dirname(__file__) + '/FBCSP_IIR_8a30.pickle')

LH_RH = R.iloc[np.where(R['Classes']=='[1, 2]')]

#N_CSP = [ [ R.iloc[ np.where((R['N CSP']==i)&(R['N Sbands']==j)) ] for j in sbands ] for i in ncomp ]

# Melhores N sub-bandas para cada um dos pares de filtros CSP (Para 4 classes um_contra_um = 6 combinações)
Step1_mean = [ [ R.iloc[ np.where((R['N CSP']==i)&(R['N Sbands']==j)) ]['Acc'].mean() for j in sbands ] for i in ncomp ] 
Step2_nsb_best = np.ravel([ np.where(Step1_mean[i] == max(Step1_mean[i])) for i in range(len(Step1_mean)) ])
Step3_acc_best = np.asarray([ Step1_mean[i][j] for i,j in zip(range(len(Step1_mean)), Step2_nsb_best) ])
Best_Sb_NComp = pd.DataFrame(
        np.concatenate([ncomp.reshape(11,1),
                        Step2_nsb_best.reshape(11,1) + 1,
                        Step3_acc_best.reshape(11,1)
                        ], axis=1), columns=['N CSP','Best N Sb','Mean Acc'])

# Melhores N sub-bandas para cada um dos pares de filtros CSP (Para 2 classes LHxRH)
Step1_mean2 = [ [ LH_RH.iloc[ np.where((LH_RH['N CSP']==i)&(LH_RH['N Sbands']==j)) ]['Acc'].mean() for j in sbands ] for i in ncomp ] 
Step2_nsb_best2 = np.ravel([ np.where(Step1_mean2[i] == max(Step1_mean2[i])) for i in range(len(Step1_mean2)) ])
Step3_acc_best2 = np.asarray([ Step1_mean2[i][j] for i,j in zip(range(len(Step1_mean2)), Step2_nsb_best2) ])
Best_Sb_NComp2 = pd.DataFrame(
        np.concatenate([ncomp.reshape(11,1),
                        Step2_nsb_best2.reshape(11,1) + 1,
                        Step3_acc_best2.reshape(11,1)
                        ], axis=1), columns=['N CSP','Best N Sb','Mean Acc'])

# Média Acc por sujeito (4 classes)
X1 = R.iloc[ np.where((R['N CSP']==10)&(R['N Sbands']==10)) ]
MeanSuj1 = np.asarray([ X1.iloc[np.where(X1['Subj']==i)]['Acc'].mean() for i in suj ])
print('Média Total: ',np.mean(MeanSuj1))

# Média Acc por sujeito (LH x RH)
X2 = LH_RH.iloc[ np.where((LH_RH['N CSP']==10)&(LH_RH['N Sbands']==10)) ]
MeanSuj2 = np.asarray([ X2.iloc[np.where(X2['Subj']==i)]['Acc'].mean() for i in suj ])
print('Média Total: ',np.mean(MeanSuj2)) # np.mean(X2['Acc'])

S1 = R.iloc[np.nonzero(R['Subj']==1)] # R.iloc[np.where(R['Subj']==1)]

#SumCost = R.iloc[np.where(R['Subj']==1) ]['Cost'].sum()
#MeanSB = np.asarray([ LH_RH.iloc[np.where(LH_RH['N Sbands']== i)].iloc[:,3:5].mean() for i in sbands])
