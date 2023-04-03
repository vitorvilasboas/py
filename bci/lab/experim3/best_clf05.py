# -*- coding: utf-8 -*-
# @author: vboas
import numpy as np
import math
import scipy.linalg as lg
import seaborn as srn
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm # utilizado na implementação de instruções padrão R
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter, butter
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression


class CSP():
    def __init__(self, n_comp):
        self.ncomp = n_comp
        self.filters_ = None
    
    def fit(self, dados, y):
        n_epocas, n_canais, n_amostras = dados.shape
        labels = np.unique(y)   
        Ca = dados[labels[0] == y,:,:]
        Cb = dados[labels[1] == y,:,:]
        covA = np.zeros((n_canais, n_canais))
        covB = np.zeros((n_canais, n_canais))
        for epoca in range(int(n_epocas/2)):
            covA += np.dot(Ca[epoca], Ca[epoca].T)
            covB += np.dot(Cb[epoca], Cb[epoca].T)
#            covA += (np.cov(Ca[epoca])/n_canais) # covariância normalizada por num canais
#            covB += (np.cov(Cb[epoca])/n_canais) # covariância normalizada por num canais
#        covA /= np.trace(covA) # np.trace(cov_1) = np.sum(np.diag(cov_1)) = soma da diagonal principal
#        covB /= np.trace(covB) # np.trace(cov_1) = np.sum(np.diag(cov_1)) = soma da diagonal principal
        [D, W] = lg.eigh(covA, covA + covB)
        ind = np.empty(n_canais, dtype=int)
        ind[::2] = np.arange(n_canais - 1, n_canais // 2 - 1, -1) 
        ind[1::2] = np.arange(0, n_canais // 2)
        W = W[:, ind]
        self.filters_ = W.T[:self.ncomp]
        return self
    
    def transform(self, dados):        
        Xt = np.asarray([np.dot(self.filters_, epoca) for epoca in dados])
        Xcsp = np.log(np.mean(Xt ** 2, axis=2))
        return Xcsp


def janelamento(dados, Fs, Ti, Tf):
    Tdica = 2
    ini = int((Ti + Tdica) * Fs)
    fim = int((Tf + Tdica) * Fs)
    janela = dados[:,:,ini:fim]
    return janela


def filtragemIIR(X, Fs, f0, fn, ordem):
    nyquist = Fs/2.
    b, a = butter(ordem, [f0/nyquist, fn/nyquist], btype='bandpass')
    XF = lfilter(b, a, X)
    return XF

def correlaciona(var_indep, var_dep): # correlação entre var_indep e var_dep 
    R = np.corrcoef(var_indep, var_dep)[0,1] 
    var_indep = var_indep.reshape(-1,1)
    modeloReg = LinearRegression()
    modeloReg.fit(var_indep, var_dep)
    R2 = modeloReg.score(var_indep, var_dep) # Calculo R2: indica quanto var_indep explica var_dep
    #previsoes = modeloReg.predict(var_indep)
    #y_interceptacao = modeloReg.intercept_
    #inclinacao = modeloReg.coef_
    return R,R2


if __name__ == '__main__':
    sujeitos = np.arange(1,10)
    classes = np.arange(1,3)
    datasets = ['T_','E_']
    n_componentes = 8
    jmin = 1
    Windows = [] # Janela: início (entre 0-3s a cada 0.5s); fim (entre 1-4 a cada 0.5s) 
    for Ji in np.arange(0, 4.5 - jmin, 0.5): 
        for Jf in np.arange(Ji + jmin, 4.5, 0.5): Windows.append([Ji,Jf,Jf-Ji])
    Windows = np.asarray(Windows)
    classifiers = ['LDA','SVM','KNN','DT','Bayes','MLP']
    FINAL = []
    for classifier in classifiers:
        AccClf = []
        for suj in sujeitos:
            AccSuj = []
            D = []
            for cl in classes: 
                dados = [ np.load('/home/vboas/datasets/BCICIV_2a/npy/A0' + str(suj) + ds + str(cl) + '.npy') for ds in datasets ]
                D.append(np.concatenate((dados[0],dados[1]))) # concatena por tipo de dataset (T ou E)  
            Z = np.concatenate((D[0], D[1])) # Vetor de entrada unico (T+E) : 144 LH + 144 RH  
            for w in range(len(Windows)): 
                ZJ = janelamento(Z, 250, Windows[w,0], Windows[w,1]) # Janelas definidas dentro do espaço em que há MI
                X = filtragemIIR(ZJ, 250, 8, 30, 5) # Filragem temporal para atenuação de ruídos e artefatos
                modeloCSP = CSP(n_componentes) # Seleção e Extração de características - Redução de dimensionalidade
                if   classifier == 'LDA': modeloCLF = LDA() # Cria o modelo de classificação - instancia o classificador
                elif classifier == 'SVM': modeloCLF = SVC(kernel="poly", C=10**(-4))
                elif classifier == 'KNN': modeloCLF = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) #minkowski e p=2 -> para usar distancia euclidiana padrão
                elif classifier == 'DT': modeloCLF = DecisionTreeClassifier(criterion='entropy', random_state=0) #max_depth = None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
                elif classifier == 'Bayes': modeloCLF = GaussianNB()
                elif classifier == 'MLP': modeloCLF = MLPClassifier(verbose=False, max_iter=10000, tol=0.0001, activation='logistic', learning_rate_init=0.001, learning_rate='invscaling',  solver='adam') #hidden_layer_sizes=(100,), 
                y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) # vetor gabarito                 
                clf = Pipeline([('CSP', modeloCSP), ('SVC', modeloCLF)]) # executa uma sequencia de processamento com um classificador no final
                cv = StratifiedShuffleSplit(10, test_size=0.2, random_state=42)
                cross_scores = cross_val_score(clf, X, y, cv=cv)                    
                AccSuj.append(np.asarray(cross_scores).mean()) # Add média validação cruzada à lista de resultados do sujeito
            AccClf.append(AccSuj) # Adiciona resultados do sujeito na matriz global de acurácias    
        AccClf = np.asarray(AccClf).T 
        WIN = pd.DataFrame(Windows, columns=['Jini','Jfim','Jdim']) # Criando dataframe Resultante
        ACC = pd.DataFrame(AccClf, columns=['S{}'.format(s) for s in sujeitos])
        FINAL.append(pd.concat([WIN,ACC], axis=1)) # result[0].iloc[:,3]

        
    # Análise e visualização dos resultados
    MClf = np.array([[np.mean(FINAL[c].iloc[:,3:].mean()), np.median(FINAL[c].iloc[:,3:]), np.mean(FINAL[c].iloc[:,3:].std())] for c in range(len(classifiers)) ])
    MClf = pd.DataFrame(MClf, columns=['Media','Mediana','Desvio']) # Medidas de Centralidade de todos os classificadores
    
    MSuj = []
    for i in range(len(FINAL)):
        MSuj.append(np.asarray([ FINAL[i]['S{}'.format(s)].median() for s in sujeitos ]).T)    
    MSuj = pd.DataFrame(np.concatenate([MSuj], axis=1).T, columns=['{}'.format(c) for c in classifiers]) # Medianas por sujeito
    #print(round(MS*100,1),'\n',round(MS.mean()*100,1))
    
    BEST = FINAL[0] # Desempenho (Acc) LDA todos sujeitos todas janelas [9,28]
    BEST_MEAN = np.asarray([BEST.iloc[:,3:].mean(), BEST.iloc[:,3:].median(), BEST.iloc[:,3:].std()]).T
    BEST_MEAN = pd.DataFrame(BEST_MEAN, columns=['Media','Mediana','Desvio']) # Medidas de Centralidade do melhor classificador (LDA)
    
    Mini_unic = []  # acc média possíveis inicios por sujeito
    Mfim_unic = []  # acc média possíveis fins por sujeito
    Mdim_unic = []  # acc média possíveis dimensões por sujeito
    best_win  = []   # melhores janelas por sujeito
    for s in range(len(sujeitos)):
        Mini_unic.append( [ BEST.iloc[Windows[:,0] == i, 3+s].mean() for i in np.unique(BEST['Jini']) ] )   
        Mfim_unic.append( [ BEST.iloc[Windows[:,1] == i, 3+s].mean() for i in np.unique(BEST['Jfim']) ] ) 
        Mdim_unic.append( [ BEST.iloc[Windows[:,2] == i, 3+s].mean() for i in np.unique(BEST['Jdim']) ] )
        idx_max = list(BEST.iloc[:,3+s]).index(max(BEST.iloc[:,3+s])) # indice da acc maxima do sujeito
        best_win.append(BEST.iloc[idx_max, [0,1,2,3+s]]) # dados da melhor janela por sujeito
    Mini_unic = np.asarray(Mini_unic).T
    Mfim_unic = np.asarray(Mfim_unic).T
    Mdim_unic = np.asarray(Mdim_unic).T
    best_win = np.asarray(best_win)
    
    # coefs correlação (R) e determinação (R2)            
    CORR_INI = pd.DataFrame([correlaciona(np.unique(BEST['Jini']), Mini_unic[:,s]) for s in range(len(sujeitos))], columns=['R','R2'])             
    CORR_FIM = pd.DataFrame([correlaciona(np.unique(BEST['Jfim']), Mfim_unic[:,s]) for s in range(len(sujeitos))], columns=['R','R2'])             
    CORR_DIM = pd.DataFrame([correlaciona(np.unique(BEST['Jdim']), Mdim_unic[:,s]) for s in range(len(sujeitos))], columns=['R','R2'])
    
    cores1 = ['olive','firebrick','orange','lime','k','purple','gray','green','c']
    cores2 = ['c','m','darkorange','green','hotpink','firebrick','peru']
    markers = ["P","s","8","d","o","p","X","h","v"]

    
    # Barras Desempenho médio de todos os classificadores
    ax = (MSuj*100).plot(figsize=(8,5), kind='bar', color=cores2)
    ax.grid(True, axis='y', **dict(ls='--', alpha=0.6))
    ax.set_xticklabels(sujeitos, rotation = 360) # va= 'baseline', ha="right",
    ax.set_yticks(np.linspace(50,100,10, endpoint=False))
    ax.set_ylim(50,100)
    ax.set_ylabel("Acurácia(%)")
    ax.set_xlabel("Sujeito")
    # ax.set_title("Desempenho Médio dos modelos de classificação por sujeito")
    plt.savefig('desempenhoClfs.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    
    # Boxplot Acurácia LDA por sujeito
    plt.figure(figsize=(10, 7), facecolor='mintcream')
    plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    plt.boxplot(np.asarray(BEST.iloc[:,3:])*100, vert = True, showfliers = True, notch = False, patch_artist = True, 
                boxprops=dict(facecolor="silver", color="gray", linewidth=1, hatch = '/'))
    plt.xlabel('Sujeito', size=14)
    plt.ylabel('Acurácia (%)', size=14)
    plt.yticks(np.arange(40, 100, step=5))
    # plt.title('Boxplot: Acurácia do classificador LDA por sujeito (MD x ME) - 28 janelas x 9 sujeitos')
    plt.savefig('boxplotLDA.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    
    # Correlação Dimensão Janelas e Acurácia Média
    plt.figure(figsize=(10, 7), facecolor='mintcream')
    plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    # plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
    plt.xlabel('Dimensão da janela (s)', size=14)
    plt.ylabel('Acurácia (%)', size=14)
    plt.yscale('linear')
    plt.yticks(np.arange(40, 100, step=5))
    plt.xticks(np.unique(BEST['Jdim']))
    for s in range(len(sujeitos)):
        plt.plot(np.unique(BEST['Jdim']), Mdim_unic[:,s]*100, color=cores1[s], lw=1)
        plt.scatter(np.unique(BEST['Jdim']), Mdim_unic[:,s]*100, color=cores1[s], facecolors=cores1[s], marker = markers[s], 
                    label=('S{}'.format(s+1)))
    plt.legend(loc='best', ncol = 3, fontsize=11)
    # plt.title('Correlação entre dimensão da janela e acurácia média LDA')
    plt.savefig('corr_dimensao.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    
    # Correlação Inicios Janelas e Acurácia Média
    plt.figure(figsize=(10, 7), facecolor='mintcream')
    plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    # plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
    plt.xlabel('Início da janela (s)', size=14)
    plt.ylabel('Acurácia (%)', size=14)
    plt.yscale('linear')
    plt.yticks(np.arange(40, 100, step=5))
    plt.xticks(np.unique(BEST['Jini']))
    for s in range(len(sujeitos)):
        plt.plot(np.unique(BEST['Jini']), Mini_unic[:,s]*100, color=cores1[s], lw=1)
        plt.scatter(np.unique(BEST['Jini']), Mini_unic[:,s]*100, color=cores1[s], facecolors=cores1[s], marker = markers[s], 
                    label=('S{}' .format(s+1)))
    plt.legend(loc='best', ncol = 3, fontsize=11)
    # plt.title('Correlação entre início da janela e acurácia média LDA')
    plt.savefig('corr_inicios.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    
    # Boxplot Melhor Janela
    plt.figure(figsize=(10, 7), facecolor='mintcream')
    plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    # plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
    plt.boxplot(best_win[:,0:2].T, vert = True, showfliers = True, notch = False, patch_artist = True, 
                showmeans=False, meanline=False, medianprops=dict(color='lightblue'),
                boxprops=dict(facecolor="lightblue", color="purple", linewidth=1, hatch = None))
    plt.xlabel('Sujeito', size=14)
    plt.ylabel('Janela (s)', size=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    # plt.figure('Figura 9 - Boxplot: Melhores janelas para cada um dos 9 sujeitos no conjunto de dados')
    plt.savefig('boxplotWIN.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    # plt.figure(figsize=(10, 7), facecolor='mintcream')
    # # plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
    # # plt.boxplot(best[:,1:].T, vert=True, showfliers=True, notch=False, patch_artist=True, showmeans=False, meanline=False, medianprops=dict(color='lightblue'),
    # #             boxprops=dict(facecolor="lightblue", color="purple", linewidth=1, hatch=None))
    # # for i in range(1,10): plt.plot(np.ones(100)*i, np.linspace(best[i-1,1]+0.15,best[i-1,2]-0.15,100), linewidth=27, color="lightblue")
    # plt.bar(np.arange(1,10), best[:,3], bottom=best[:,1], color="lightblue", width=0.4, edgecolor='navy', linewidth=1)
    # plt.grid(axis='y', **dict(ls='--', alpha=0.6))
    # plt.xlabel('Sujeito', size=14)
    # plt.ylabel('Janela (s)', size=14)
    # plt.ylim((0.3,4.2))
    # plt.yticks((np.arange(0.5,4.5,0.5)), fontsize=12)
    # plt.xticks(np.arange(1,10),fontsize=12)
    # # plt.figure('Figura 9 - Boxplot: Janela com melhor de sempenho de generalização por sujeito')
    # # plt.savefig('/home/vboas/cloud/overleaf/Artigo_DM/boxplotWIN.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    

