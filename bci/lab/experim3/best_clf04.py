# -*- coding: utf-8 -*-
# @author: vboas
import numpy as np
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
            #covA += np.dot(Ca[epoca], Ca[epoca].T)
            #covB += np.dot(Cb[epoca], Cb[epoca].T)
            covA += (np.cov(Ca[epoca])/n_canais) # covariância normalizada por num canais
            covB += (np.cov(Cb[epoca])/n_canais) # covariância normalizada por num canais
        covA /= np.trace(covA) # np.trace(cov_1) = np.sum(np.diag(cov_1)) = soma da diagonal principal
        covB /= np.trace(covB) # np.trace(cov_1) = np.sum(np.diag(cov_1)) = soma da diagonal principal
        
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


if __name__ == '__main__':
    sujeitos = np.arange(1,10)
    classes = np.arange(1,3)
    datasets = ['T_','E_']
    n_componentes = 6
    jmin = 1
    Windows = [] # Janela: início (entre 0-3s a cada 0.5s); fim (entre 1-4 a cada 0.5s) 
    for Ji in np.arange(0, 4.5 - jmin, 0.5): 
        for Jf in np.arange(Ji + jmin, 4.5, 0.5): Windows.append([Ji,Jf,Jf-Ji])
    Windows = np.asarray(Windows)
    classifiers = ['LDA', 'KNN']#, 'SVM', 'Bayes', 'Tree'] #, 'RNA'
    FINAL = []
    for classifier in classifiers:
        AccClf = []
        for suj in sujeitos:
            AccSuj = []
            D = []
            for cl in classes: 
                dados = [ np.load('/home/vboas/devto/datasets/BCICIV_2a/npy/A0' + str(suj) + ds + str(cl) + '.npy') for ds in datasets ]
                D.append(np.concatenate((dados[0],dados[1]))) # concatena por tipo de dataset (T ou E)  
            Z = np.concatenate((D[0], D[1])) # Vetor de entrada unico (T+E) : 144 LH + 144 RH  
            for w in range(len(Windows)): 
                Z = janelamento(Z, 250, Windows[w,0], Windows[w,1]) # Janelas definidas dentro do espaço em que há MI
                X = filtragemIIR(Z, 250, 8, 30, 5) # Filragem temporal para atenuação de ruídos e artefatos
                modeloCSP = CSP(n_componentes) # Seleção e Extração de características - Redução de dimensionalidade
                if classifier == 'LDA': modeloCLF = LDA() # Cria o modelo de classificação - instancia o classificador
                elif classifier == 'SVM': modeloCLF = SVC(kernel="poly", C=10**(-4)) # SVM
                elif classifier == 'RNA': modeloCLF = MLPClassifier(verbose=False, max_iter=1000, tol=0.00001, activation='logistic', learning_rate_init=0.01) # RNA MLP
                # elif classifier == 'RNA': modeloCLF = MLPClassifier(verbose=False, max_iter=10000, tol=0.00001, activation='logistic', learning_rate_init=0.0001) # RNA MLP
                elif classifier == 'Bayes': modeloCLF = GaussianNB()
                elif classifier == 'Tree': modeloCLF = DecisionTreeClassifier(criterion='entropy', random_state=0)
                elif classifier == 'KNN': modeloCLF = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
                y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) #vetor gabarito de classe
#                kfold = StratifiedShuffleSplit(10, test_size=0.2, random_state=42) # shuffle e random_state(semente geradora) garantem a aleatoriedade na escolha dos dados de terino e teste
#                cross_scores = []
#                for indice_treinamento, indice_teste in kfold.split(X, y):
#                    modeloCSP.fit(X[indice_treinamento], y[indice_treinamento])
#                    XTcsp = modeloCSP.transform(X[indice_treinamento])
#                    XEcsp = modeloCSP.transform(X[indice_teste])
#                    modeloCLF.fit(XTcsp, y[indice_treinamento])
#                    previsoes = modeloCLF.predict(XEcsp)
#                    precisao = accuracy_score(y[indice_teste], previsoes)
#                    cross_scores.append(precisao)                  
                clf = Pipeline([('CSP', modeloCSP), ('SVC', modeloCLF)]) # executa uma sequencia de processamento com um classificador no final
                cv = StratifiedShuffleSplit(10, test_size=0.2, random_state=42)
                cross_scores = cross_val_score(clf, X, y, cv=cv)                    
                AccSuj.append(np.asarray(cross_scores).mean()) # Add média validação cruzada à lista de resultados do sujeito
            AccClf.append(AccSuj) # Adiciona resultados do sujeito na matriz global de acurácias    
        AccClf = np.asarray(AccClf).T 
        WIN = pd.DataFrame(Windows, columns=['Jini','Jfim','Jdim']) # Criando dataframe Resultante
        ACC = pd.DataFrame(AccClf, columns=['S{}'.format(s) for s in sujeitos])
        FINAL.append(pd.concat([WIN,ACC], axis=1)) # result[0].iloc[:,3]
        
    
        
#        #ini_unicos = np.unique(RES['Jini']) # extraindo alfabeto de inicios e fins 
#        #fins_unicos = np.unique(RES['Jfim']) # outra foma de extrair
#        #dim_unicas = np.unique(RES['Jdim'])
#        # md_suj = np.asarray([np.array([ACC.iloc[:,s].mean(), ACC.iloc[:,s].median(), ACC.iloc[:,s].std()]) for s in range(len(sujeitos))]) # dados acurácia media todos sujeitos
#        
#        md_ini = [] # media inicios janelas todos sujeitos
#        md_fim = [] # media fins janelas todos sujeitos
#        md_dim = [] # media dimensões janelas todos sujeitos
#        best_win = [] # dados melhores janelas todos sujeitos
#        for s in range(len(sujeitos)):
#            idx_max_acc = list(ACC.iloc[:,s]).index(max(ACC.iloc[:,s]))
#            best_win.append(RES.iloc[idx_max_acc, [0,1,2,s+3]]) # dados da melhor janela por sujeito
#            md_ini.append([AccGlobal[Windows[:,0] == wi, s].mean() for wi in ini_unicos]) # acc média de cada inicio por sujeito  
#            md_fim.append([AccGlobal[Windows[:,1] == wi, s].mean() for wi in fins_unicos]) # acc média de cada fim por sujeito 
#            md_dim.append([AccGlobal[Windows[:,2] == wi, s].mean() for wi in dim_unicas]) # acc média de cada dimensão por sujeito
#        best_win = np.asarray(best_win)
#        md_ini = np.asarray(md_ini).T
#        md_fim = np.asarray(md_fim).T
#        md_dim = np.asarray(md_dim).T
#        
#        sumario_classificador = np.zeros((len(sujeitos),5))
#        for s in range(len(sujeitos)):
#            sumario_classificador[s,0] = 100 + cont # representa o classificador (100=LDA, 106=KNN)
#            sumario_classificador[s,1] = s+1
#            for j in range(3): sumario_classificador[s,j+2] = round(md_suj[s,j]*100,2)
#
#        ## Modelo de Regressão linear simples [Cálculo do coeficiente de correlação (R) e do Coeficiente de determinação (R2)]
#        modelo_regressao = LinearRegression()
#        reg_previsoes = []
#        coeficientes = np.zeros((len(sujeitos),5))
#        for s in range(len(sujeitos)):
#            # coef_cor = np.corrcoef(dim_unicas, medias_dim_win_suj[:,s])[0,1] # correlação entre Dimenses de janelas e respectivas taxas de classificação média para cada dimensão possível
#            # var_indep = dim_unicas.reshape(-1,1)
#            # var_dep = medias_dim_win_suj[:,s]
#            coef_cor = np.corrcoef(ini_unicos, md_ini[:,s])[0,1] # correlação entre Incios de janelas e respectivas taxas de classificação média 
#            var_indep = ini_unicos.reshape(-1,1)
#            var_dep = md_ini[:,s]
#            
#            modelo_regressao.fit(var_indep, var_dep)
#            reg_previsoes.append(modelo_regressao.predict(var_indep))
#            
#            y_interceptacao = modelo_regressao.intercept_
#            inclinacao = modelo_regressao.coef_
#            coef_r2 = modelo_regressao.score(var_indep, var_dep) # Calculo R2 - indicador do quanto a variavel independente explica a variavel dependente
#    
#            coeficientes[s] = [s+1,coef_cor,coef_r2,y_interceptacao,inclinacao]
#        
#        coeficientes = pd.DataFrame(coeficientes, columns=['Suj','R','R2','y_icept','inclin']) 
#        # print(coeficientes['R'])
#
#        # paletas de cores para uso em gráficos
#        cores1 = ['olive','dimgray','darkorange','firebrick','lime','k','peru','c','purple']
#        cores2 = ['c','m','orange','firebrick','green','gray','hotpink']
#        
#        #### GRÁFICOS DO CLASSIFICADOR AQUI ####
#        
#        sumarioFull.append(sumario_classificador) # id_classifieras = 100..105
#        cont += 1
#    
#    ## Comentar se execução de único modelo
#    sumarioFull = np.asarray(sumarioFull)
#    for c in range(len(sumarioFull)):
#        print(round(sumarioFull[c,:,2].mean(),2),'\t', round(sumarioFull[c,:,3].mean(),2),'\t', 
#              round(sumarioFull[c,:,4].mean(),2))
#    
#    sf = sumarioFull[0]
#    for i in range(1,len(sumarioFull)): sf = np.concatenate([sf,sumarioFull[i]])
#    #plt.figure(figsize=(10, 6), facecolor='mintcream')
#    #plt.bar(sf[:,1],sf[:,3], color=cores2, tick_label=sf[:,0])
#    
#    resultado = pd.DataFrame(sf, columns=['Classifier','Sujeito','MediaAcc','MedianaAcc','DesvioAcc'])
#    
#    raux = np.asarray(resultado["Classifier"])
#    raux = [str(int(i)) for i in raux ]
#    for i in range(len(classifiers)):
#        raux = np.where(raux==str(i+100), classifiers[i], raux)
#    
#    resultado["Classifier"] = raux
#    
#    ax = resultado['MediaAcc'].plot(figsize=(8,5), kind='bar')
#    ax.set_xticklabels(resultado[['Classifier','Sujeito']])

