# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import math
import time
import mne
import numpy as np
import scipy.linalg as lg
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter, butter
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import seaborn as srn
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



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
    folder = "/home/vboas/master_tools/datasets/bcicIV_2a/"
    Fs = 250
    janela_min = 1
    Tmin, Tmax = 2, 6 # Start trial= 0s ; Start dica= 2s ; End MI= 6s ; # Inicio e Fim da IM respectivamente (em segundos)
    
    # Parâmetros de filtragem temporal-espectral e espacial
    filtro = 'IIR' # FFT ou IIR
    ordem = 5
    freq0 = 8
    freqN = 30
    
    # Parâmetros de processamento (modelo)
    n_componentes = 6
    perc_teste = 0.7
    k_folds = 10
    classifier = 'LDA' # LDA, SVM, RNA, Bayes, Tree, KNN
    
    # Matriz de janelas onde a primeira coluna contém o início de cada janela abrangendo o intervao entre 0s e (4 - janela_min)s a cada 0.5s,
    # a segunda coluna contém o fim de cada janela a cada 0.5s no intervalo entre (0+janela_min)s e 4s,
    # e a terceira coluna contém a dimensão de cada janela.
    Windows = [[],[],[]]
    for Ji in np.arange(0, 4.5 - janela_min, 0.5): 
        for Jf in np.arange(Ji + janela_min, 4.5, 0.5):
            Windows[0].append(Ji)
            Windows[1].append(Jf) 
            Windows[2].append(Jf-Ji)
    
    
    ## lista para armazenar todo o histórico de performance da classificação de cada uma das 28 janelas para cada um dos 9 sujeitos
    AccGlobal = []
    
    Windows = np.asarray(Windows).T
    
    for suj in sujeitos:
        S = []
        AccSuj = []
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
            clf = Pipeline([('CSP', modeloCSP), ('SVC', modeloCLF)]) # executa uma sequencia de processamento com um classificador no final
            cv = StratifiedShuffleSplit(k_folds, test_size=0.7, random_state=42)
            cross_scores = cross_val_score(clf, X, y, cv=cv)
            print('({}-{}) Validação cruzada: {} {} '.format(Windows[w,0], Windows[w,1], round(cross_scores.mean() * 100, 2), round(cross_scores.std() *100, 2)))
            
            # Add média da validação cruzada à lista de resultados do sujeito
            AccSuj.append(cross_scores.mean())
            
            ### Abordagem 2: Validação única, segregando dataset em dados de treino e validação
            
            ## Amostragem e segregação em treino e validação 100% arbitrária
            # XTreino = np.concatenate([X[:48],X[144:192]]) # 30%
            # XTeste = np.concatenate([X[48:144],X[192:]]) # 70%
            # neT, ncT, naT = XTreino.shape
            # neE, ncE, naE = XTeste.shape
            # ytreino = np.concatenate([np.zeros(int(neT/2)), np.ones(int(neT/2))]) #vetor gabarito de classe
            # yteste = np.concatenate([np.zeros(int(neE/2)), np.ones(int(neE/2))]) #vetor gabarito de classe
            
            ## Amostragem Simples e segregação randômica em treino e validação 
            # amostras = np.random.choice[a = [0, 1], size = 288, replace = True, p = [1 - perc_teste, perc_teste]] # 70% teste, 30% treino
            # XTreino = X[amostras == 0]
            # XTeste = X[amostras == 1]
            
            ## Amostragem Estratificada (Ideal)
            XTreino, XTeste, ytreino, yteste = train_test_split(X, y, test_size = perc_teste, stratify = y) # separa o dataset em treino e validaçãos
            
            modeloCSP.fit(XTreino, ytreino) # treina o CSP
            XTcsp = modeloCSP.transform(XTreino)
            XVcsp = modeloCSP.transform(XTeste)
            
            modeloCLF.fit(XTcsp, ytreino) # treina o classificador a partir do dataset de treino
            classificacao = modeloCLF.predict(XVcsp) # realiza a predição a partir do dataset de avaiação
            acuracia = np.mean(classificacao == yteste) # obtém a taxa de acerto através da média = razão de classificações corretas em relação ao total de classificações
            
            # print('Validação: {}' .format(round(acuracia * 100, 2)))
        
        ## Adiciona resultados do sujeito na matriz global de acurácias
        AccGlobal.append(AccSuj)  
        
    AccGlobal = np.asarray(AccGlobal).T
    
    
    ## Criando dataframe Resultante
    header_suj = ['S1','S2','S3','S4','S5','S6','S7','S8','S9']
    header_win = ['Jini','Jfim','Jdim']
    dfw = pd.DataFrame(Windows, columns=header_win)
    dfacc = pd.DataFrame(AccGlobal, columns=header_suj)
    RES = pd.concat([dfw,dfacc], axis=1)
    
    # extraindo alfabeto de inicios e fins
    ini_unicos = np.unique(RES['Jini']) 
    fim_unicos = list(set(RES['Jfim'])) # outra foma de extrair
    dim_unicas = np.unique(RES['Jdim'])
    
    ## Imprime a média total da média, mediana e desvio padrão da taxa de acerto do classificador
    print(round(((RES.iloc[:,3:]).mean()).mean()*100,2), round(((RES.iloc[:,3:]).median()).mean()*100,2), round(((RES.iloc[:,3:]).std()).mean()*100,2))
    
    medias_acc_suj = [] 
    best_win = []
    for s in header_suj:  
        medias_acc_suj.append([RES[s].mean(), RES[s].median(), RES[s].std()]) # Armazenando a média, mediana e desvio padrão de cada sujeito
        best_win.append(RES.iloc[list(RES[s]).index(max(RES[s])), [0,1,2,]]) # Armazenando dados da janelas com melhor acurácia de cada sujeito
        
        media_wini = []
        for wi in ini_unicos:
            media_wini.append(RES.iloc[RES['Jini'] == wi, s].mean())
            
    medias_acc_suj = np.asarray(medias_acc_suj)
    best_win = np.asarray(best_win)

     
    
    
    
    
    
    
    
    
    '''
    medias_ini = []
    
    melhor = []
    for i in range(len(sujeitos)):
        media_sujeito = []
        for j in np.arange(0, 4.5 - janela_min, 0.5):
            media_sujeito.append(AccGlobal[AccGlobal[:,9] == j, i].mean())
            # print(i+1, j)
        print(max(AccGlobal[:,i]), list(AccGlobal[:,i]).index(max(AccGlobal[:,i])))
        
        medias_ini.append(media_sujeito)
        inicio.append(AccGlobal[list(AccGlobal[:,i]).index(max(AccGlobal[:,i])), 9])
        fim.append(AccGlobal[list(AccGlobal[:,i]).index(max(AccGlobal[:,i])), 10])
    
    medias_ini = np.asarray(medias_ini)
    
    
    melhor.append(inicio)
    melhor.append(fim)
    melhor = np.asarray(melhor).T
    plt.boxplot(medias_ini.T)
    
    indice = AccGlobal[:,10] == fins[i]
    
    for i in range(len(fins)):
        for s in range(len(sujeitos)):
            plt.scatter(AccGlobal[indice,9], AccGlobal[indice,s],  label = fins[i])
    plt.legend(loc = 'lower right')
    
    
    # gráfico dispersão acurácia_média x inicio da janela
    for i in range(len(sujeitos)):
        plt.scatter(inicios, medias_ini[i,:], color = 'blue', facecolors = 'none', marker = 'o')
        plt.plot(inicios, medias_ini[i,:])
    
    
    
    
    testeee = []
    for i in range(len(sujeitos)):
        ind5 = np.where(AccGlobal[:,i] >= AccGlobal[:,i].mean())
        testeee.append(AccGlobal[ind5,9])
    testeee = np.asarray(testeee)
    plt.boxplot(testeee)
    
    
    srn.boxplot(data = AccGlobal[:,0:9])
    plt.boxplot(AccGlobal[:,0:9]*100)
    
    
    sujeito1 = []
    sujeito1.append(AccGlobal[:,0])
    sujeito1.append(AccGlobal[:,9])
    
    plt.boxplot(np.asarray(sujeito1).T)
    
    print(list(AccGlobal[:,2]).index(max(AccGlobal[:,2])))
    
    
    
    # plt.boxplot(AccGlobal[:,1], vert = False, showfliers = False, notch = True, patch_artist = True)
    
    for i in range(len(AccGlobal[0])): 
        fig, ax = plt.subplots()
        pos = np.array(i)
        bp = ax.boxplot(AccGlobal[:,i]) 
    plt.setp(bp['whiskers'], color='k', linestyle='-')
    plt.setp(bp['fliers'], markersize=3.0)
    plt.show()
        # print(i+1, round(AccGlobal[:,i].mean()*100,2), round(AccGlobal[:,i].std()*100,2))
        
    '''