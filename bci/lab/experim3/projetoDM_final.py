# -*- coding: utf-8 -*-
# @author: vboas
import math
import time
import mne
import numpy as np
import scipy.linalg as lg
import seaborn as srn
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm # utilizado na implementação de instruções padrão R
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter, butter
from scipy.fftpack import fft
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression


def corrigeNaN(dados):
    for canal in range(dados.shape[0] - 1):
        this_chan = dados[canal]
        dados[canal] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(dados[canal])
        mediaCanal = np.nanmean(dados[canal])
        dados[canal, mask] = mediaCanal
    return dados


def nanCleaner(epocas):
    # Remove NaN por interpolação
    for ep in epocas:
        for i in range(ep.shape[0]):
            bad_idx = np.isnan(ep[i, :])
            ep[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], ep[i, ~bad_idx])
    return epocas


def carregaBase(path):
    ### Carrega Dataset usando pacote MNE
    mne.set_log_level('WARNING','DEBUG')
    raw = mne.io.read_raw_gdf(path)
    raw.load_data()
    dados_suj = raw.get_data() # 
    eventos_suj = raw.find_edf_events()
    return dados_suj, eventos_suj


def normRotulos(ev, ds, folder, suj):
    rotulos = ev[0][:,2]
    rotulos = np.where(rotulos==1, 1023, rotulos) # Rejected trial
    rotulos = np.where(rotulos==2, 768, rotulos) # Start trial t=0
    rotulos = np.where(rotulos==3, 1072, rotulos) # Eye movements / Unknown
    
    if ds=='T': # if Training dataset (A0sT.gdf)
        
        rotulos = np.where(rotulos==8, 277, rotulos) # Idling EEG (eyes closed) 
        rotulos = np.where(rotulos==9, 276, rotulos) # Idling EEG (eyes open) 
        rotulos = np.where(rotulos==10, 32766, rotulos) # Start of a new run/segment (after a break) 
        rotulos = np.where(rotulos==4, 769, rotulos) # LH (classe 1) 
        rotulos = np.where(rotulos==5, 770, rotulos) # RH (classe 2) 
        rotulos = np.where(rotulos==6, 771, rotulos) # Foot (classe 3)
        rotulos = np.where(rotulos==7, 772, rotulos) # Tongue (classe 4)
        
        for i in range(0, len(rotulos)): 
            if rotulos[i]==768: # rotula [1 a 4] o inicio da trial...
                if rotulos[i+1] == 1023: rotulos[i] = rotulos[i+2] - rotulos[i]
                else: rotulos[i] = rotulos[i+1] - rotulos[i] # a partir da proxima tarefa [ 1 para 769, 2 para 770... ]
        
    else: # if Evaluate dataset (A0sE.gdf)
        
        rotulos = np.where(rotulos==5, 277, rotulos) # Idling EEG (eyes closed) 
        rotulos = np.where(rotulos==6, 276, rotulos) # Idling EEG (eyes open) 
        rotulos = np.where(rotulos==7, 32766, rotulos) # Start of a new run/segment (after a break)
        
        # Carregando Rótulos verdadeiros para uso em datasets de validação (E)
        trueLabels = np.ravel(loadmat(folder + 'true_labels/A0' + str(suj) + 'E.mat')['classlabel'])
        
        idx4 = np.where(rotulos==4)
        rotulos[idx4] = trueLabels + 768
        
        idx768 = np.where(rotulos==768)
        rotulos[idx768] = trueLabels
    
    ev[0][:,2] = rotulos
    
    return ev


def extraiEpocas(data, eventos, classes, Fs, Tmin, Tmax):
    
    rotulos = eventos[0][:,2]
    cond = False
    for i in range(len(classes)): cond += (rotulos == classes[i])
    # cond é um vetor, cujo indice contém True se a posição correspondente em rotulos contém 1, 2, 3 ou 4
    idx = np.where(cond)[0] # contém os 288 indices que correspondem ao carimbo de um das quatro classes 
    
    stamps = eventos[0][idx, 0] #contém os sample_stamp(posições) relacionadas ao inicio das 288 tentativas
    win_begin = stamps + (math.floor(Tmin * Fs)) # vetor que marca as amostras que iniciam cada época com IM após a dica
    win_end = stamps + (math.floor(Tmax * Fs))  # vetor que marca as amostras que findam cada época com IM

    n_epocas = len(stamps)
    n_canais = data.shape[0]
    n_amostras = (Tmax-Tmin)*Fs
    
    epocas = np.zeros([n_epocas, n_canais, n_amostras])
    labels = rotulos[idx] # vetor que contém os indices das 288 épocas das 4 classes
    epocas_incompletas = []
    
    for i in range(n_epocas):
        epoca = data[:, win_begin[i]:win_end[i]]
        if epoca.shape[1] == n_amostras: # Checa se a época está completa/integra
            epocas[i, :, :] = epoca
        else:
            print('Época incompleta')
            epocas_incompletas.append(i)
    
    labels = np.delete(labels, epocas_incompletas)
    epocas = np.delete(epocas, epocas_incompletas, axis=0)
    
    # organiza épocas separando-as por classe em uma nova dimensão
    ep = []
    for lb in classes:
        idx = np.where(labels==lb)
        ep.append(epocas[idx])
    
    return ep


def janelamento(dados, Fs, Ti, Tf):
    ini = int(Ti * Fs)
    fim = int(Tf * Fs)
    janela = dados[:,:,ini:fim]
    return janela


def filtragemFFT(dados, Fs, f0, fn):
    nyquist = Fs/2.
    bin0 = int(f0 * (Fs/nyquist))  # para fl = 8 bin0 = 15
    binN = int(fn * (Fs/nyquist)) # para fl = 8 bin0 = 15
    FFTz = fft(dados)
    REAL = np.transpose(np.real(FFTz)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
    IMAG = np.transpose(np.imag(FFTz)[:,:,bin0:binN], (2, 0, 1)) #transpoe para intercalar
    Filtrado = list(np.itertools.chain.from_iterable(zip(IMAG, REAL))) #intercalando
    Filtrado = np.transpose(Filtrado, (1, 2, 0)) # retorna ao formato original      
    return Filtrado


def filtragemIIR(dados, Fs, f0, fn, ordem):
    nyquist = Fs/2.
    b, a = butter(ordem, [f0/nyquist, fn/nyquist], btype='bandpass')
    Filtrado = lfilter(b, a, dados)
    return Filtrado


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


if __name__ == '__main__':
    # Parâmetros gerais
    sujeitos = np.arange(1,10)
    classes = np.arange(1,5)
    datasets = ['T','E']
    folder = "/home/vboas/devto/datasets/BCICIV_2a/"
    Fs = 250
    janela_min = 1
    Tmin, Tmax = 2, 6 # Start trial= 0s ; Start dica= 2s ; End MI= 6s ; # Inicio e Fim da IM respectivamente (em segundos)
    
    # Parâmetros de filtragem temporal-espectral e espacial
    filtro = 'IIR' # FFT ou IIR
    ordem = 5
    freq0 = 8
    freqN = 30
    
    # Parâmetros de processamento (modelo)
    n_componentes = 10
    perc_teste = 0.7
    k_folds = 10
    classifier = 'LDA' # KNN, SVM, RNA, Bayes, Tree, LDA
    
    # Matriz de janelas onde a primeira coluna contém o início de cada janela abrangendo o intervao entre 0s e (4 - janela_min)s a cada 0.5s,
    # a segunda coluna contém o fim de cada janela a cada 0.5s no intervalo entre (0+janela_min)s e 4s,
    # e a terceira coluna contém a dimensão de cada janela.
    Windows = [[],[],[]]
    for Ji in np.arange(0, 4.5 - janela_min, 0.5): 
        for Jf in np.arange(Ji + janela_min, 4.5, 0.5):
            Windows[0].append(Ji)
            Windows[1].append(Jf) 
            Windows[2].append(Jf-Ji)
    
    Windows = np.asarray(Windows).T
    
    ''' ## descomentar bloco abaixo para avaliação de todos os modelos de classificação
    cont = 0
    sumarioFull = []
    classifiers = ['KNN', 'SVM', 'RNA', 'Bayes', 'Tree', 'LDA']
    for classifier in classifiers:
    '''
    
    ## lista para armazenar todo o histórico de performance da classificação de cada uma das 28 janelas para cada um dos 9 sujeitos
    AccGlobal = []
    MatrizGlobal = []

    for suj in sujeitos:
        S = []
        AccSuj = []
        MatrizSuj = []
        for ds in datasets: 
            # Carregmento de dados brutos e eventos
            dados, eventos = carregaBase(folder + 'A0' + str(suj) + ds + '.gdf') 
            # eventos_old = list(eventos[0][:,2]) 
            
            # Extração de variáveis (canais) relevantes 
            dados = dados[range(22)] 
            
            # Normalização de valores NaN
            dados = corrigeNaN(dados) 
            
            # Rotulando corretamente os eventos conforme descrição da competição
            eventos = normRotulos(eventos, ds, folder, suj) 
            
            # Extração de todas as 288 épocas (72 por classe)
            epocas = extraiEpocas(dados, eventos, classes, Fs, Tmin, Tmax) # [4x72x22x1000]
            
            # normalização/correção de valores NaN nas épocas
            epocas = nanCleaner(epocas)
            
            # Adiciona as epocas extraidas ao vetor de dados master do sujeito (S)
            S.append(epocas)
        
        # transpoe Z para concatenar/juntar as epocas de T e E do sujeito, já que todas já estão devidamente rotuladas
        S = np.concatenate([np.transpose(S[0],(1,0,2,3)), np.transpose(S[1],(1,0,2,3))])
        
        # retorna Z ao formato original
        S = np.transpose(S, (1,0,2,3))
        ## Aqui temos um vetor único de dimensões (4 x 144 x 22 x 1000) = (n_classes x n_epocas x n_canais x n_amostras), 
        ## onde S[0]=classe 1(lh); S[1]=classe 2(rh); S[2]=classe 3(f); S[3]=classe 4(t)
        
        # separando e concatenando dados de duas classes para processamento        
        Z = np.concatenate([S[0], S[1]]) # ...nesse caso: lh + rh
        
        for w in range(len(Windows)):
            # Janelamento e redimensionamento das amostras conforme intervalo [Ji:Jf]
            ZJ = janelamento(Z, Fs, Windows[w,0], Windows[w,1]) #Parâmetros que determinam o inicio e o fim da janela amostral dentro do período em que há IM
            
            # Filragem temporal para atenuação de ruídos e artefatos
            if filtro == 'IIR': X = filtragemIIR(ZJ, Fs, freq0, freqN, ordem)
            elif filtro == 'FFT': X = filtragemFFT(ZJ, Fs, freq0, freqN)
            
            # Seleção e Extração de características - Redução de dimensionalidade
            n_epocas, n_canais, n_amostras = X.shape
            y = np.concatenate([np.zeros(int(n_epocas/2)), np.ones(int(n_epocas/2))]) #vetor gabarito de classe
            modeloCSP = CSP(n_componentes)
            
             # Cria o modelo de classificação - instancia o classificador
            if classifier == 'LDA': modeloCLF = LDA()
            elif classifier == 'SVM': modeloCLF = SVC(kernel="linear", C=10**(-4)) # SVM
            elif classifier == 'RNA': modeloCLF = MLPClassifier(verbose=False, max_iter=1000, tol=0.00001, activation='logistic', learning_rate_init=0.01) # RNA MLP
            # elif classifier == 'RNA': modeloCLF = MLPClassifier(verbose=False, max_iter=10000, tol=0.00001, activation='logistic', learning_rate_init=0.0001) # RNA MLP
            elif classifier == 'Bayes': modeloCLF = GaussianNB()
            elif classifier == 'Tree': modeloCLF = DecisionTreeClassifier(criterion='entropy', random_state=0)
            elif classifier == 'KNN': modeloCLF = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
            
            ### Abrodagem 1: Validação Cruzada
            ## Abordagem compacta
            '''
            clf = Pipeline([('CSP', modeloCSP), ('SVC', modeloCLF)]) # executa uma sequencia de processamento com um classificador no final
            cv = StratifiedShuffleSplit(k_folds, test_size=0.7, random_state=42)
            # cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 3)
            cross_scores = cross_val_score(clf, X, y, cv=cv)
            '''
            
            ## Abordagem com criação de matrizes de confusão
            # kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
            kfold = StratifiedShuffleSplit(k_folds, test_size=0.7, random_state=42) # shuffle e random_state(semente geradora) garantem a aleatoriedade na escolha dos dados de terino e teste
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
            MatrizSuj.append(np.mean(matrizes, axis = 0))                  
            cross_scores = np.asarray(cross_scores)
            
            # print('({}-{}) Validação cruzada: {} {} '.format(Windows[w,0], Windows[w,1], round(cross_scores.mean() * 100, 2), round(cross_scores.std() *100, 2)))
            
            # Add média da validação cruzada à lista de resultados do sujeito
            AccSuj.append(cross_scores.mean())
            
            '''
            ### Abordagem 2: Validação única, segregando dataset em dados de treino e validação
            
            ## Amostragem e segregação em treino e validação 100% arbitrária
            #XTreino = np.concatenate([X[:48],X[144:192]]) # 30%
            #XTeste = np.concatenate([X[48:144],X[192:]]) # 70%
            XTreino = np.concatenate([X[:72],X[144:216]]) # 50%
            XTeste = np.concatenate([X[72:144],X[216:]]) # 50%
            
            neT, ncT, naT = XTreino.shape
            neE, ncE, naE = XTeste.shape
            ytreino = np.concatenate([np.zeros(int(neT/2)), np.ones(int(neT/2))]) #vetor gabarito de classe
            yteste = np.concatenate([np.zeros(int(neE/2)), np.ones(int(neE/2))]) #vetor gabarito de classe
            
            ## Amostragem Simples e segregação randômica em treino e validação 
            # amostras = np.random.choice[a = [0, 1], size = 288, replace = True, p = [1 - perc_teste, perc_teste]] # 70% teste, 30% treino
            # XTreino = X[amostras == 0]
            # XTeste = X[amostras == 1]
            '''
            '''
            ## Amostragem Estratificada (Ideal)
            XTreino, XTeste, ytreino, yteste = train_test_split(X, y, test_size = perc_teste, stratify = y) # separa o dataset em treino e validaçãos
            
            modeloCSP.fit(XTreino, ytreino) # treina o CSP
            XTcsp = modeloCSP.transform(XTreino)
            XVcsp = modeloCSP.transform(XTeste)
            
            modeloCLF.fit(XTcsp, ytreino) # treina o classificador a partir do dataset de treino
            classificacao = modeloCLF.predict(XVcsp) # realiza a predição a partir do dataset de avaiação
            #acuracia = np.mean(classificacao == yteste) # obtém a taxa de acerto através da média = razão de classificações corretas em relação ao total de classificações
            acuracia = accuracy_score(yteste, classificacao)
            AccSuj.append(acuracia)
            
            MatrizSuj.append(confusion_matrix(yteste, classificacao))
            # print('Validação: {}' .format(round(acuracia * 100, 2)))
            '''
        ## Adiciona resultados do sujeito na matriz global de acurácias
        AccGlobal.append(AccSuj)  
        
        MatrizGlobal.append(MatrizSuj)
        
    AccGlobal = np.asarray(AccGlobal).T
    
    MatrizGlobal1 = np.asarray(MatrizGlobal).T
        
    
    # --------------------------------------- #
    ## ANÁLISE DOS RESULTADOS
    # --------------------------------------- #
    ## Formatando e gerando objetos para análise
    
    ## Criando dataframe Resultante
    WIN = pd.DataFrame(Windows, columns=['Jini','Jfim','Jdim'])
    ACC = pd.DataFrame(AccGlobal, columns=['S1','S2','S3','S4','S5','S6','S7','S8','S9'])
    RES = pd.concat([WIN,ACC], axis=1)
    
    # extraindo alfabeto de inicios e fins
    ini_unicos = np.unique(WIN['Jini']) 
    fins_unicos = list(set(WIN['Jfim'])) # outra foma de extrair
    dim_unicas = np.unique(WIN['Jdim'])
    
    ## Apresenta a média total da média, mediana e desvio padrão da taxa de acerto do classificador
    # print(round(((RES.iloc[:,3:]).mean()).mean()*100,2), round(((RES.iloc[:,3:]).median()).mean()*100,2), round(((RES.iloc[:,3:]).std()).mean()*100,2))
    # print(round(AccGlobal[:,i].mean()*100,2), round(AccGlobal[:,i].median()*100,2), round(AccGlobal[:,i].std()*100,2))
    
    medias_acc_suj = [] # dados acurácia media todos sujeitos
    best_win = [] # dados melhores janelas todos sujeitos
    medias_ini_win_suj = [] # media inicios janelas todos sujeitos
    medias_fim_win_suj = [] # media fins janelas todos sujeitos
    medias_dim_win_suj = [] # media dimensões janelas todos sujeitos
    for s in range(len(sujeitos)):  
        medias_acc_suj.append([ACC.iloc[:,s].mean(), ACC.iloc[:,s].median(), ACC.iloc[:,s].std()]) # Armazenando a média, mediana e desvio padrão de cada sujeito
        # idx_max_acc = list(AccGlobal[:,s]).index(max(AccGlobal[:,s]))
        idx_max_acc = list(ACC.iloc[:,s]).index(max(ACC.iloc[:,s]))
        best_win.append(RES.iloc[idx_max_acc, [0,1,2,s+3]]) # Armazenando dados da janelas com melhor acurácia de cada sujeito
        
        md = []
        for wi in ini_unicos: md.append(AccGlobal[Windows[:,0] == wi, s].mean()) # Armazenando médias de cada um dos 7 inicios de janela para cada sujeito
        medias_ini_win_suj.append(md)
        
        md = []
        for wi in fins_unicos: md.append(AccGlobal[Windows[:,1] == wi, s].mean()) # Armazenando médias de cada um dos 7 fins de janela para cada sujeito
        medias_fim_win_suj.append(md) 
        
        md = []
        for wi in dim_unicas: md.append(AccGlobal[Windows[:,2] == wi, s].mean()) # Armazenando médias de cada um das 7 dimensões de janela para cada sujeito
        medias_dim_win_suj.append(md)
    
    medias_acc_suj = np.asarray(medias_acc_suj)
    best_win = np.asarray(best_win)
    medias_ini_win_suj = np.asarray(medias_ini_win_suj).T
    medias_fim_win_suj = np.asarray(medias_fim_win_suj).T
    medias_dim_win_suj = np.asarray(medias_dim_win_suj).T
    
    sumario_classificador = np.zeros((len(sujeitos),5)) #,4
    for s in range(len(sujeitos)):
        # sumario_classificador[s,0] = 100 + cont # representa o classificador (100=LDA, 106=KNN) (comentar se somente LDA)
        sumario_classificador[s,0] = s+1 #sumario_classificador[s,1] (se somente LDA)
        for j in range(3):
            sumario_classificador[s,j+1] = round(medias_acc_suj[s,j]*100,2) #sumario_classificador[s,j+2] (se somente LDA)
            medias_acc_suj[s,j] = round(medias_acc_suj[s,j]*100,2)
        print(s+1,'\t',medias_acc_suj[s,0],'\t',medias_acc_suj[s,1],'\t',medias_acc_suj[s,2])
    
    # print('mean\t',round(medias_acc_suj[:,0].mean(),2),'\t', round(medias_acc_suj[:,1].mean(),2),'\t', round(medias_acc_suj[:,2].mean(),2))  
    
    ## Modelo de Regressão linear simples [Cálculo do coeficiente de correlação (R) e do Coeficiente de determinação (R2)]
    modelo_regressao = LinearRegression()
    reg_previsoes = []
    coeficientes = np.zeros((len(sujeitos),5))
    for s in range(len(sujeitos)):
        # coef_cor = np.corrcoef(dim_unicas, medias_dim_win_suj[:,s])[0,1] # correlação entre Dimenses de janelas e respectivas taxas de classificação média para cada dimensão possível
        # var_indep = dim_unicas.reshape(-1,1)
        # var_dep = medias_dim_win_suj[:,s]
        coef_cor = np.corrcoef(ini_unicos, medias_ini_win_suj[:,s])[0,1] # correlação entre Incios de janelas e respectivas taxas de classificação média 
        var_indep = ini_unicos.reshape(-1,1)
        var_dep = medias_ini_win_suj[:,s]
        
        modelo_regressao.fit(var_indep, var_dep)
        reg_previsoes.append(modelo_regressao.predict(var_indep))
        
        y_interceptacao = modelo_regressao.intercept_
        inclinacao = modelo_regressao.coef_
        coef_r2 = modelo_regressao.score(var_indep, var_dep) # Calculo R2 - indicador do quanto a variavel independente explica a variavel dependente

        coeficientes[s] = [s+1,coef_cor,coef_r2,y_interceptacao,inclinacao]
    
    coeficientes = pd.DataFrame(coeficientes, columns=['Suj','R','R2','y_icept','inclin'])
    
    # print(round(coeficientes['R'], 2))
    # print(round(coeficientes['R2'], 2))
    
    
    # --------------------------------------- #
    ## REPRESENTAÇÃO GRÁFICA DOS RESULTADOS
    # --------------------------------------- #
    
    # paletas de cores para uso em gráficos
    cores_suj = ['olive','dimgray','darkorange','firebrick','lime','k','peru','c','purple']
    cores_fim = ['c','m','orange','firebrick','green','gray','hotpink']
    
    
    
    # -----------------------------------------------------------------------
    plt.figure(1, facecolor='mintcream')
    plt.subplots(figsize=(10, 6), facecolor='mintcream')
    # plt.title('Figura 3 - Boxplot: Acurácia do classificador LDA por sujeito (MD x ME) - 28 janelas x 9 sujeitos')
    plt.boxplot(AccGlobal*100, vert = True, showfliers = True, notch = False, patch_artist = True, 
                boxprops=dict(facecolor="dimgray", color="purple", linewidth=1, hatch = '/'))
    plt.xlabel('Sujeito', size=14)
    plt.ylabel('Acurácia (%)', size=14)
    plt.yticks(np.arange(40, 100, step=2.5))
    
    # -----------------------------------------------------------------------
    plt.figure(2, facecolor='mintcream')
    plt.subplots(figsize=(10, 6), facecolor='mintcream')
    # plt.title('Figura 4 - Dimensão da janela em relação à acurácia de classificação LDA - Plotagem única todos os sujeito')
    for s in range(len(sujeitos)):
        plt.subplot(1,1,1)
        plt.scatter(dim_unicas, medias_dim_win_suj[:,s]*100, color=cores_suj[s], facecolors = cores_suj[s], marker = 'o', 
                    label=('Sujeito {}' .format(s+1)))
        plt.plot(dim_unicas, medias_dim_win_suj[:,s]*100, color=cores_suj[s])
        plt.xlabel('Dimensão da janela (s)', fontsize=14)
        plt.ylabel('Acurácia (%)', size=14)
        plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
        plt.yscale('linear', fontsize=12)
        plt.yticks(np.arange(50, 100, step=2.5), fontsize=14)
        plt.legend(loc='upper left', ncol = 3, fontsize=12)
    
    # -----------------------------------------------------------------------
    modelo_regressao = LinearRegression()
    plt.figure(3, facecolor='mintcream')
    plt.subplots(figsize=(10, 6), facecolor='mintcream')
    # plt.title('Figura 5 - Gráfico de Dispersão - Correlação/Regressão entre a dimensão da janela e a acurácia de classificação para cada sujeito - Plotagem em forma de matriz')  
    for s in range(len(sujeitos)):
        plt.subplot(3,3,s+1)
        var_indep = dim_unicas.reshape(-1,1)
        var_dep = medias_dim_win_suj[:,s]
        modelo_regressao.fit(var_indep, var_dep)
        plt.scatter(var_indep, var_dep*100, facecolors = cores_suj[s], marker = 'o', label=('Sujeito {}' .format(s+1)))
        plt.plot(var_indep, modelo_regressao.predict(var_indep)*100, color='red', linewidth=1.5, linestyle='-')
        plt.xlabel('Dimensão da janela (s)', fontsize=12)
        plt.ylabel('Acurácia (%)', fontsize=12)
        plt.legend(loc='upper left', fontsize=12)
        
    # -----------------------------------------------------------------------
    plt.figure(4, facecolor='mintcream')
    plt.subplots(figsize=(10, 6), facecolor='mintcream')
    # plt.title('Figura 6 - Início da janela em relação à acurácia média de classificação LDA - Plotagem única todos os sujeitos')
    for s in range(len(sujeitos)):
        plt.subplot(1,1,1)
        plt.scatter(ini_unicos, medias_ini_win_suj[:,s]*100, color=cores_suj[s], facecolors = cores_suj[s], marker = 'o', 
                    label=('Sujeito {}' .format(s+1)))
        plt.plot(ini_unicos, medias_ini_win_suj[:,s]*100, color=cores_suj[s])
        plt.xlabel('Início da janela (s)', fontsize=14)
        plt.ylabel('Acurácia (%)', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
        plt.yscale('linear', fontsize=12)
        plt.yticks(np.arange(50, 100, step=2.5), fontsize=13)
        plt.legend(loc='upper right', ncol = 3, fontsize=12)
        
    # -----------------------------------------------------------------------
    modelo_regressao = LinearRegression()
    plt.figure(5, facecolor='mintcream')
    plt.subplots(figsize=(10, 6), facecolor='mintcream')
    # plt.title('Figura 7 - Gráfico de dispersão - Correlação/Regressão entre o início da janela e a acurácia de classificação para cada sujeito - Plotagem em forma de matriz')
    for s in range(len(sujeitos)):
        plt.subplot(3,3,s+1)
        var_indep = ini_unicos.reshape(-1,1)
        var_dep = medias_ini_win_suj[:,s]
        modelo_regressao.fit(var_indep, var_dep)
        plt.scatter(var_indep, var_dep*100, facecolors = cores_suj[s], marker = 'o', label=('Sujeito {}' .format(s+1)))
        plt.plot(var_indep, modelo_regressao.predict(var_indep)*100, color='red', linewidth=1.5, linestyle='-')
        plt.xlabel('Início da janela (s)', fontsize=12)
        plt.ylabel('Acurácia (%)', fontsize=12)
        plt.legend(loc='upper right', fontsize=12)
    
    # -----------------------------------------------------------------------
    plt.figure(6, facecolor='mintcream')
    plt.subplots(figsize=(10, 6), facecolor='mintcream')
    # plt.figure('Figura 9 - Boxplot: Melhores janelas para cada um dos 9 sujeitos no conjunto de dados')
    plt.boxplot(best_win[:,0:2].T, vert = True, showfliers = True, notch = False, patch_artist = True, 
                showmeans=False, meanline=False, medianprops=dict(color='lightblue'),
                boxprops=dict(facecolor="lightblue", color="purple", linewidth=1, hatch = None))
    plt.xlabel('Sujeito', fontsize=14)
    plt.ylabel('Janela (s)', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', linewidth=1, color='gainsboro')
    
    # -----------------------------------------------------------------------
    plt.figure(7, facecolor='mintcream')
    plt.subplots(figsize=(10, 6), facecolor='mintcream')
    # plt.figure('Figura 10 - Gráfico scatter - Inícios em relação aos fins das 28 janelas e o impacto dessa relação na acurácia do classificador - Plotagem  em forma de matriz') 
    for s in range(len(sujeitos)):
        for i in range(len(fins_unicos)):
            indice = []
            indice = Windows[:,1] == fins_unicos[i]
            plt.subplot(3,3,s+1)
            plt.scatter(Windows[indice,0], AccGlobal[indice,s]*100, color=cores_fim[i])
            # plt.plot(Windows[indice,0], AccGlobal[indice,s]*100, color=cores_fim[i])
            plt.xlabel('Início da janela (s)', fontsize=12)
            plt.ylabel('Acurácia (%)', fontsize=12)
            plt.yticks()
            plt.legend(loc = 'upper right', ncol = 7, fontsize=12, title='Sujeito {}' .format(s+1))       
    # plt.legend(fins_unicos, loc = 'lower center', ncol = 7, fontsize=10, title='Fim da janela') # gerar separadamente


    '''    ## descomentar bloco abaixo para avaliação de todos os modelos de classificação
        # 100=KNN, 101=SVM, 102=RNA, 103=Bayes, 104=Tree, 105=LDA
        sumarioFull.append(sumario_classificador)
        cont += 1
        
    sumarioFull = np.asarray(sumarioFull)
    
    plt.figure(1, facecolor='mintcream')
    plt.subplots(figsize=(10, 6), facecolor='mintcream')
    # plt.figure('Figura 2 - Gráfico de dispersão - Comparativo entre os modelos de classificação testados)
    for c in range(len(sumarioFull)):
        plt.subplot(1,1,1)
        plt.scatter(sumarioFull[c,:,1], sumarioFull[c,:,3], color=cores_fim[c], facecolors = cores_fim[c], marker = '.')
        plt.plot(sumarioFull[c,:,1], sumarioFull[c,:,3], color=cores_fim[c], linewidth=2, label=('{}' .format(classifiers[c])))
        plt.xlabel('Sujeito', fontsize=14)
        plt.ylabel('Mediana da Acurácia entre as janelas (%)', fontsize=14)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.legend(loc='upper left', ncol = 2, fontsize=14, title='Modelo Classificador')
           
    for c in range(len(sumarioFull)):
        print(round(sumarioFull[c,:,2].mean(),2),'\t', 
              round(sumarioFull[c,:,3].mean(),2),'\t', 
              round(sumarioFull[c,:,4].mean(),2))
    '''