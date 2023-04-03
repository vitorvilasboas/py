# -*- coding: utf-8 -*-
from numpy import ceil, concatenate, log, mean, ones, ravel, sign, std, sum, zeros
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sys import path
from time import time

path.append('../general')

from filtering_proj import filtering
from get_step import get_step
from load_data import load_data
from mnecsp import CSP
from windowing import windowing


def get_acc(subject, classes, args):

    fl, fh, m, n_components, Clog, n_bands, numtaps, atraso = args
    m = int(m)
    n_bands = int(n_bands)

    # Carregando os dados da memória. X contém dados de treinamento e validação de duas classes
    X = load_data(subject, classes, atraso)
    # print('Raw: ', len(X), len(X[0]), len(X[0][0]), len(X[0][0][0]), len(X[0][0][0][0]))

    # Todos os dados são janelados entre 2.5 e 4.5 segundos. Fornecendo também o tempo inicial nesses arquivos que
    # é igual a 2 segundos e a frequência de amostragem, 250 Hz
    X = windowing(X, 2, 2.5, 4.5, 250, atraso) # baseado no inicio do protocolo experimental e não na instrução
    # print('Windowed: ', len(X), len(X[0]), len(X[0][0]), len(X[0][0][0]), len(X[0][0][0][0]))

    # Set data format for CSP
    # Aqui eu junto os dados de treinamento e validação cada um em um tensor (XT e XV respectivamente).
    # Cada um tem as dimensões (ne, nc, nt), e estão organizados de forma que as épocas correspondentes à primeira
    # classe venham primeiro e depois as da segunda classe
    # ne = Número de épocas (144 no total, 72 para cada classe)
    # nc = Número de canais (22)
    # nt = Número de amostras no tempo (500, 2 segundos em 250 Hz)
    XT = concatenate(X[0])
    XV = concatenate(X[1])
    # print('Concatened(T+E): ', XT.shape)

    # Definindo um vetor de tamanho 144, onde a primeira metade são todos 0 e a segunda metade são todos 1.
    # Isso vai ser usado como vetor de classes (gabarito)
    y = concatenate([zeros(72), ones(72)])

    # Filtering
    # Aqui é feita a projeção dos sinais na base de senos e cossenos. São fornecidos as frequências limite (fl, fh) e
    # o número de senos/cossenos na base (m). Os tensores na saída são da forma (ne, nc, 2*m) e
    # os coeficientes de seno/cosseno ficam intercalados de forma que o vetor fique ordenado de acordo com a frequencia.
    # Detalhes de como isso é feito estão na função filtering
    XT = filtering(XT, fl=fl, fh=fh, m=m)
    XV = filtering(XV, fl=fl, fh=fh, m=m)
    # print('Filtered: ', XT.shape)

    # Divide sub-bands
    # Aqui são definidas algumas variaveis usadas para a divisão de bandas
    n_bins = 2 * m  # n_bins = Número de bins de frequencia (2*m) ou (XT.shape[2])
    overlap = 2  # overlap = Overlap entre as bandas, nesse caso é de 50% (1/overlap)
    # step = Intervalo de coeficientes entre duas bandas, resultado do número de bins, bandas e do overlap definido.
    step = get_step(n_bins, n_bands, overlap)  # Mais detalhes na função get_step
    size = step * overlap  # size = Número de coeficientes em cada banda - add conversão int()?

    # Aqui as bandas são separadas. Cada banda indexada por i é um pedaço do tensor original de dimensão(ne,nc,2*m).
    # Após isso são obtidos n_bands bandas, cada banda sendo representada por um tensor de dimensão (ne, nc, size)
    XT = [XT[:, :, i*step:i*step+size] for i in range(n_bands)]
    XV = [XV[:, :, i*step:i*step+size] for i in range(n_bands)]
    # print('Bandwing: ', len(XT), len(XT[0]), len(XT[0][0]), len(XT[0][0][0]))

    # CSP
    # Vários (n_bands) CSPs são definidos e treinados com os dados do conjunto de treinamento (XT).
    # XT[i] é o tensor pra banda i, e o vetor y de classes é o mesmo pra todos
    csp = [CSP(n_components=n_components) for i in range(n_bands)]
    for i in range(n_bands):
        csp[i].fit(XT[i], y)

    # Depois do treinamento os dados são transformados, cada XT[i] e XV[i] vira uma matriz de forma (ne, n_components).
    # Mais detalhes do treinamento e transformação estão na classe CSP
    XT_CSP = [csp[i].transform(XT[i]) for i in range(n_bands)]
    XV_CSP = [csp[i].transform(XV[i]) for i in range(n_bands)]
    # print('Pos Treina CSP: ', len(XT_CSP), len(XT_CSP[0]), len(XT_CSP[0][0]))

    # LDA
    # Definindo uma matriz de dimensão (ne, n_bands)
    SCORE_T = zeros((144, n_bands))
    SCORE_V = zeros((144, n_bands))

    # Da mesma forma que o CSP, um LDA é treinado para cada banda
    clf = [LinearDiscriminantAnalysis() for i in range(n_bands)]
    for i in range(n_bands):
        clf[i].fit(XT_CSP[i], y)

        # Para cada banda, esse LDA treinado transforma os dados de treinamento e validação.
        # Relembrando, não é feita uma classificação 'dura', o resultado contínuo do LDA é mantido para construir um
        # vetor de características de dimensão (n_bands)
        SCORE_T[:, i] = ravel(clf[i].transform(XT_CSP[i]))
        SCORE_V[:, i] = ravel(clf[i].transform(XV_CSP[i]))
    # print('Pos LDA: ', len(SCORE_T), len(SCORE_T[0]))

    # Bayesian meta-classifier
    # Aqui é treinado o meta-classificador bayesiano. Os dados de treinamento são separados entre os pertencentes
    # à classe 0 e classe 1. Para cada classe, calculamos a média e o desvio padrão entre as épocas, de forma que
    # os vetores sejam de dimensão n_bands
    SCORE_T0 = SCORE_T[y == 0, :]
    m0 = mean(SCORE_T0, axis=0)
    std0 = std(SCORE_T0, axis=0)
    # print('Score 0e1: ', len(SCORE_T0), len(SCORE_T0[0]))

    SCORE_T1 = SCORE_T[y == 1, :]
    m1 = mean(SCORE_T1, axis=0)
    std1 = std(SCORE_T1, axis=0)

    # p0 e p1 representam uma distribuição normal de médias m0 e m1, e desvio padrão std0 e std1
    p0 = norm(m0, std0)
    p1 = norm(m1, std1)

    # Os scores são aplicados na função de densidade de probabilidade de cada classe e calculamos a razão entre elas.
    # Um resultado positivo de META_SCORE_T indica que uma época é melhor representada pela função p0,
    # já que p0.pdf(SCORE_T) > p1.pdf(SCORE_T). No caso contrário META_SCORE_T é negativo
    META_SCORE_T = log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
    META_SCORE_V = log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))

    # SVM on top of the meta-classifier
    # A partir dos meta-scores um SVM linear é treinado e aplicado nos dados de validação
    svc = SVC(kernel="linear", C=10**Clog)
    svc.fit(META_SCORE_T, y)
    ans = svc.predict(META_SCORE_V)

    # ans == y retorna 1 quando a classe estimada é correta (igual a y) e 0 quando é incorreta.
    # Então a média desse vetor é igual a taxa de acertos do classificador
    return mean(ans == y)


if __name__ == "__main__":

    subject = 1
    classes = [1,2]
    
    fl = 0
    fh = 51
    m = 100
    n_components = 6
    Clog = -4
    n_bands = 33
    
    args = (fl, fh, m, n_components, Clog, n_bands)
    
    t0 = time()
    acc_test = get_acc(subject, classes, args)
    print(time()-t0)
    
    print('Test accuracy: ' + str(acc_test*100))
