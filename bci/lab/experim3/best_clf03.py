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
    
    janela_min = 1
    Windows = [[],[],[]] # Janela: início (entre 0-3s a cada 0.5s); fim (entre 1-4 a cada 0.5s) 
    for Ji in np.arange(0, 4.5 - janela_min, 0.5): 
        for Jf in np.arange(Ji + janela_min, 4.5, 0.5):
            Windows[0].append(Ji)
            Windows[1].append(Jf) 
            Windows[2].append(Jf-Ji)
    Windows = np.asarray(Windows).T
    
    cont = 0
    sumarioFull = []
    classifiers = ['LDA', 'KNN', 'SVM', 'Bayes', 'Tree'] #, 'RNA'
    for classifier in classifiers:
        AccGlobal = []
        for suj in sujeitos:
            S = []
            AccSuj = []
            for cl in classes: 
                dados = [ np.load('/home/vboas/devto/datasets/BCICIV_2a/npy/A0' + str(suj) + ds + str(cl) + '.npy') for ds in datasets ]
                S.append(np.concatenate((dados[0],dados[1]))) # concatena por tipo de dataset (T ou E)  
            Z = np.concatenate((S[0], S[1])) # Vetor de entrada unico (T+E) : 144 LH + 144 RH  
            
            for w in range(len(Windows)): 
                ZJ = janelamento(Z, 250, Windows[w,0], Windows[w,1]) # Janelas definidas dentro do espaço em que há MI
                
                X = filtragemIIR(ZJ, 250, 8, 30, 5) # Filragem temporal para atenuação de ruídos e artefatos
                
                modeloCSP = CSP(n_componentes) # Seleção e Extração de características - Redução de dimensionalidade
                
                # Cria o modelo de classificação - instancia o classificador
                if classifier == 'LDA': modeloCLF = LDA()
                elif classifier == 'SVM': modeloCLF = SVC(kernel="poly", C=10**(-4)) # SVM
                elif classifier == 'RNA': modeloCLF = MLPClassifier(verbose=False, max_iter=1000, tol=0.00001, activation='logistic', learning_rate_init=0.01) # RNA MLP
                # elif classifier == 'RNA': modeloCLF = MLPClassifier(verbose=False, max_iter=10000, tol=0.00001, activation='logistic', learning_rate_init=0.0001) # RNA MLP
                elif classifier == 'Bayes': modeloCLF = GaussianNB()
                elif classifier == 'Tree': modeloCLF = DecisionTreeClassifier(criterion='entropy', random_state=0)
                elif classifier == 'KNN': modeloCLF = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
                
                y = np.concatenate([np.zeros(int(len(X)/2)), np.ones(int(len(X)/2))]) #vetor gabarito de classe
                                   
                ## Validação Cruzada: Abordagem com criação de matrizes de confusão
                # kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
                kfold = StratifiedShuffleSplit(10, test_size=0.2, random_state=42) # shuffle e random_state(semente geradora) garantem a aleatoriedade na escolha dos dados de terino e teste
                cross_scores = []
                matrizes = []
                for indice_treinamento, indice_teste in kfold.split(X, y):
                    # print('Índice treinamento: ', indice_treinamento, 'Índice teste: ', indice_teste)
                    modeloCSP.fit(X[indice_treinamento], y[indice_treinamento])
                    XTcsp = modeloCSP.transform(X[indice_treinamento])
                    XEcsp = modeloCSP.transform(X[indice_teste])
                    modeloCLF.fit(XTcsp, y[indice_treinamento])
                    previsoes = modeloCLF.predict(XEcsp)
                    precisao = accuracy_score(y[indice_teste], previsoes)
                    matrizes.append(confusion_matrix(y[indice_teste], previsoes))
                    cross_scores.append(precisao)                  
                
                ## Validação Cruzada: Abordagem compacta
                #clf = Pipeline([('CSP', modeloCSP), ('SVC', modeloCLF)]) # executa uma sequencia de processamento com um classificador no final
                #cv = StratifiedShuffleSplit(10, test_size=0.2, random_state=42)
                #cross_scores = cross_val_score(clf, X, y, cv=cv)
                                     
                AccSuj.append(np.asarray(cross_scores).mean()) # Add média validação cruzada à lista de resultados do sujeito
            
            AccGlobal.append(AccSuj) # Adiciona resultados do sujeito na matriz global de acurácias    
        
        AccGlobal = np.asarray(AccGlobal).T
        
        # Criando dataframe Resultante
        WIN = pd.DataFrame(Windows, columns=['Jini','Jfim','Jdim'])
        ACC = pd.DataFrame(AccGlobal, columns=['S1','S2','S3','S4','S5','S6','S7','S8','S9'])
        RES = pd.concat([WIN,ACC], axis=1)
        # e.g. RES.iloc[:,3:].median()
         
        # extraindo alfabeto de inicios e fins
        ini_unicos = np.unique(WIN['Jini']) 
        fins_unicos = list(set(WIN['Jfim'])) # outra foma de extrair
        dim_unicas = np.unique(WIN['Jdim'])
        
        #média, mediana e desvio padrão por sujeito
        md_suj = np.asarray([np.array([ACC.iloc[:,s].mean(), ACC.iloc[:,s].median(), ACC.iloc[:,s].std()]) for s in range(len(sujeitos))]) # dados acurácia media todos sujeitos
        
        md_ini = [] # media inicios janelas todos sujeitos
        md_fim = [] # media fins janelas todos sujeitos
        md_dim = [] # media dimensões janelas todos sujeitos
        best_win = [] # dados melhores janelas todos sujeitos
        for s in range(len(sujeitos)):
            idx_max_acc = list(ACC.iloc[:,s]).index(max(ACC.iloc[:,s]))
            best_win.append(RES.iloc[idx_max_acc, [0,1,2,s+3]]) # dados da melhor janela por sujeito
            
            # acc média de cada inicio por sujeito 
            md_ini.append([AccGlobal[Windows[:,0] == wi, s].mean() for wi in ini_unicos]) 
            # acc média de cada fim por sujeito
            md_fim.append([AccGlobal[Windows[:,1] == wi, s].mean() for wi in fins_unicos])  
            # acc média de cada dimensão por sujeito
            md_dim.append([AccGlobal[Windows[:,2] == wi, s].mean() for wi in dim_unicas]) 
        
        best_win = np.asarray(best_win)
        md_ini = np.asarray(md_ini).T
        md_fim = np.asarray(md_fim).T
        md_dim = np.asarray(md_dim).T
        
        sumario_classificador = np.zeros((len(sujeitos),5))
        for s in range(len(sujeitos)):
            sumario_classificador[s,0] = 100 + cont # representa o classificador (100=LDA, 106=KNN)
            sumario_classificador[s,1] = s+1
            for j in range(3): sumario_classificador[s,j+2] = round(md_suj[s,j]*100,2)

        ## Modelo de Regressão linear simples [Cálculo do coeficiente de correlação (R) e do Coeficiente de determinação (R2)]
        modelo_regressao = LinearRegression()
        reg_previsoes = []
        coeficientes = np.zeros((len(sujeitos),5))
        for s in range(len(sujeitos)):
            # coef_cor = np.corrcoef(dim_unicas, medias_dim_win_suj[:,s])[0,1] # correlação entre Dimenses de janelas e respectivas taxas de classificação média para cada dimensão possível
            # var_indep = dim_unicas.reshape(-1,1)
            # var_dep = medias_dim_win_suj[:,s]
            coef_cor = np.corrcoef(ini_unicos, md_ini[:,s])[0,1] # correlação entre Incios de janelas e respectivas taxas de classificação média 
            var_indep = ini_unicos.reshape(-1,1)
            var_dep = md_ini[:,s]
            
            modelo_regressao.fit(var_indep, var_dep)
            reg_previsoes.append(modelo_regressao.predict(var_indep))
            
            y_interceptacao = modelo_regressao.intercept_
            inclinacao = modelo_regressao.coef_
            coef_r2 = modelo_regressao.score(var_indep, var_dep) # Calculo R2 - indicador do quanto a variavel independente explica a variavel dependente
    
            coeficientes[s] = [s+1,coef_cor,coef_r2,y_interceptacao,inclinacao]
        
        coeficientes = pd.DataFrame(coeficientes, columns=['Suj','R','R2','y_icept','inclin']) 
        # print(coeficientes['R'])

        # paletas de cores para uso em gráficos
        cores1 = ['olive','dimgray','darkorange','firebrick','lime','k','peru','c','purple']
        cores2 = ['c','m','orange','firebrick','green','gray','hotpink']
        
        #### GRÁFICOS DO CLASSIFICADOR AQUI ####
        
        sumarioFull.append(sumario_classificador) # id_classifieras = 100..105
        cont += 1
    
    ## Comentar se execução de único modelo
    sumarioFull = np.asarray(sumarioFull)
    for c in range(len(sumarioFull)):
        print(round(sumarioFull[c,:,2].mean(),2),'\t', round(sumarioFull[c,:,3].mean(),2),'\t', 
              round(sumarioFull[c,:,4].mean(),2))
    
    sf = sumarioFull[0]
    for i in range(1,len(sumarioFull)): sf = np.concatenate([sf,sumarioFull[i]])
    #plt.figure(figsize=(10, 6), facecolor='mintcream')
    #plt.bar(sf[:,1],sf[:,3], color=cores2, tick_label=sf[:,0])
    
    resultado = pd.DataFrame(sf, columns=['Classifier','Sujeito','MediaAcc','MedianaAcc','DesvioAcc'])
    
    raux = np.asarray(resultado["Classifier"])
    raux = [str(int(i)) for i in raux ]
    for i in range(len(classifiers)):
        raux = np.where(raux==str(i+100), classifiers[i], raux)
    
    resultado["Classifier"] = raux
    
    ax = resultado['MediaAcc'].plot(figsize=(8,5), kind='bar')
    ax.set_xticklabels(resultado[['Classifier','Sujeito']])

    # plt.xlabel(rfinal[:,0], fontsize=14)
    #plt.legend(loc='upper left', ncol = 2, fontsize=14, title='Modelo Classificador')
    #plt.ylabel('Mediana da Acurácia entre as janelas (%)', fontsize=14)
    #plt.yticks(fontsize=13)
    #plt.xticks(fontsize=13)


#    plt.figure(1, facecolor='mintcream')
#    plt.subplots(figsize=(10, 6), facecolor='mintcream')
#    # plt.figure('Figura 2 - Gráfico de dispersão - Comparativo entre os modelos de classificação testados)
#    for c in range(len(sumarioFull)):
#        plt.subplot(1,1,1)
#        plt.scatter(sumarioFull[c,:,1], sumarioFull[c,:,3], color=cores2[c], facecolors = cores_fim[c], marker = '.')
#        plt.plot(sumarioFull[c,:,1], sumarioFull[c,:,3], color=cores2[c], linewidth=2, label=('{}' .format(classifiers[c])))
#        plt.xlabel('Sujeito', fontsize=14)
#        plt.ylabel('Mediana da Acurácia entre as janelas (%)', fontsize=14)
#        plt.yticks(fontsize=13)
#        plt.xticks(fontsize=13)
#        plt.legend(loc='upper left', ncol = 2, fontsize=14, title='Modelo Classificador')


# =============================================================================
#        # --------------------------------------- #
#        ## REPRESENTAÇÃO GRÁFICA DOS RESULTADOS
#        # --------------------------------------- #
#        # -----------------------------------------------------------------------
#        plt.figure(1, facecolor='mintcream')
#        plt.subplots(figsize=(10, 6), facecolor='mintcream')
#        # plt.title('Figura 3 - Boxplot: Acurácia do classificador LDA por sujeito (MD x ME) - 28 janelas x 9 sujeitos')
#        plt.boxplot(AccGlobal*100, vert = True, showfliers = True, notch = False, patch_artist = True, 
#                    boxprops=dict(facecolor="dimgray", color="purple", linewidth=1, hatch = '/'))
#        plt.xlabel('Sujeito', size=14)
#        plt.ylabel('Acurácia (%)', size=14)
#        plt.yticks(np.arange(40, 100, step=2.5))
#        
#        # -----------------------------------------------------------------------
#        plt.figure(2, facecolor='mintcream')
#        plt.subplots(figsize=(10, 6), facecolor='mintcream')
#        # plt.title('Figura 4 - Dimensão da janela em relação à acurácia de classificação LDA - Plotagem única todos os sujeito')
#        for s in range(len(sujeitos)):
#            plt.subplot(1,1,1)
#            plt.scatter(dim_unicas, medias_dim_win_suj[:,s]*100, color=cores_suj[s], facecolors = cores_suj[s], marker = 'o', 
#                        label=('Sujeito {}' .format(s+1)))
#            plt.plot(dim_unicas, medias_dim_win_suj[:,s]*100, color=cores_suj[s])
#            plt.xlabel('Dimensão da janela (s)', fontsize=14)
#            plt.ylabel('Acurácia (%)', size=14)
#            plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
#            plt.yscale('linear')
#            plt.yticks(np.arange(50, 100, step=2.5))
#            plt.legend(loc='upper left', ncol = 3, fontsize=12)
#        
#        # -----------------------------------------------------------------------
#        modelo_regressao = LinearRegression()
#        plt.figure(3, facecolor='mintcream')
#        plt.subplots(figsize=(10, 6), facecolor='mintcream')
#        # plt.title('Figura 5 - Gráfico de Dispersão - Correlação/Regressão entre a dimensão da janela e a acurácia de classificação para cada sujeito - Plotagem em forma de matriz')  
#        for s in range(len(sujeitos)):
#            plt.subplot(3,3,s+1)
#            var_indep = dim_unicas.reshape(-1,1)
#            var_dep = medias_dim_win_suj[:,s]
#            modelo_regressao.fit(var_indep, var_dep)
#            plt.scatter(var_indep, var_dep*100, facecolors = cores_suj[s], marker = 'o', label=('Sujeito {}' .format(s+1)))
#            plt.plot(var_indep, modelo_regressao.predict(var_indep)*100, color='red', linewidth=1.5, linestyle='-')
#            plt.xlabel('Dimensão da janela (s)', fontsize=12)
#            plt.ylabel('Acurácia (%)', fontsize=12)
#            plt.legend(loc='upper left', fontsize=12)
#            
#        # -----------------------------------------------------------------------
#        plt.figure(4, facecolor='mintcream')
#        plt.subplots(figsize=(10, 6), facecolor='mintcream')
#        # plt.title('Figura 6 - Início da janela em relação à acurácia média de classificação LDA - Plotagem única todos os sujeitos')
#        for s in range(len(sujeitos)):
#            plt.subplot(1,1,1)
#            plt.scatter(ini_unicos, medias_ini_win_suj[:,s]*100, color=cores_suj[s], facecolors = cores_suj[s], marker = 'o', 
#                        label=('Sujeito {}' .format(s+1)))
#            plt.plot(ini_unicos, medias_ini_win_suj[:,s]*100, color=cores_suj[s])
#            plt.xlabel('Início da janela (s)', fontsize=14)
#            plt.ylabel('Acurácia (%)', fontsize=14)
#            plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
#            plt.yscale('linear')
#            plt.yticks(np.arange(50, 100, step=2.5))
#            plt.legend(loc='upper right', ncol = 3, fontsize=12)
#            
#        # -----------------------------------------------------------------------
#        modelo_regressao = LinearRegression()
#        plt.figure(5, facecolor='mintcream')
#        plt.subplots(figsize=(10, 6), facecolor='mintcream')
#        # plt.title('Figura 7 - Gráfico de dispersão - Correlação/Regressão entre o início da janela e a acurácia de classificação para cada sujeito - Plotagem em forma de matriz')
#        for s in range(len(sujeitos)):
#            plt.subplot(3,3,s+1)
#            var_indep = ini_unicos.reshape(-1,1)
#            var_dep = medias_ini_win_suj[:,s]
#            modelo_regressao.fit(var_indep, var_dep)
#            plt.scatter(var_indep, var_dep*100, facecolors = cores_suj[s], marker = 'o', label=('Sujeito {}' .format(s+1)))
#            plt.plot(var_indep, modelo_regressao.predict(var_indep)*100, color='red', linewidth=1.5, linestyle='-')
#            plt.xlabel('Início da janela (s)', fontsize=12)
#            plt.ylabel('Acurácia (%)', fontsize=12)
#            plt.legend(loc='upper right', fontsize=12)
#        
#        # -----------------------------------------------------------------------
#        plt.figure(6, facecolor='mintcream')
#        plt.subplots(figsize=(10, 6), facecolor='mintcream')
#        # plt.figure('Figura 9 - Boxplot: Melhores janelas para cada um dos 9 sujeitos no conjunto de dados')
#        plt.boxplot(best_win[:,0:2].T, vert = True, showfliers = True, notch = False, patch_artist = True, 
#                    showmeans=False, meanline=False, medianprops=dict(color='lightblue'),
#                    boxprops=dict(facecolor="lightblue", color="purple", linewidth=1, hatch = None))
#        plt.xlabel('Sujeito', fontsize=14)
#        plt.ylabel('Janela (s)', fontsize=14)
#        plt.yticks(fontsize=12)
#        plt.xticks(fontsize=12)
#        plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
#        
#        # -----------------------------------------------------------------------
#        plt.figure(7, facecolor='mintcream')
#        plt.subplots(figsize=(10, 6), facecolor='mintcream')
#        # plt.figure('Figura 10 - Gráfico scatter - Inícios em relação aos fins das 28 janelas e o impacto dessa relação na acurácia do classificador - Plotagem  em forma de matriz') 
#        for s in range(len(sujeitos)):
#            for i in range(len(fins_unicos)):
#                indice = []
#                indice = Windows[:,1] == fins_unicos[i]
#                plt.subplot(3,3,s+1)
#                plt.scatter(Windows[indice,0], AccGlobal[indice,s]*100, color=cores_fim[i])
#                # plt.plot(Windows[indice,0], AccGlobal[indice,s]*100, color=cores_fim[i])
#                plt.xlabel('Início da janela (s)', fontsize=12)
#                plt.ylabel('Acurácia (%)', fontsize=12)
#                plt.yticks()
#                plt.legend(loc = 'upper right', ncol = 7, fontsize=12, title='Sujeito {}' .format(s+1))       
#        # plt.legend(fins_unicos, loc = 'lower center', ncol = 7, fontsize=10, title='Fim da janela') # gerar separadamente
#        
#        
# =============================================================================