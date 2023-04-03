# -*- coding: utf-8 -*-
from sys import stdout
from time import time
from matplotlib.pyplot import plot
from numpy import *
from scipy.linalg import eigh, pinv
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from matplotlib import pyplot


def load_data(SUBJECT, classes):
    folder = "D:/c4_2a/epocas"
    X = [[], []]
    set = ['T_', 'E_']
    for i in range(2):
        for j in range(2):
            path = folder + '/A0' + str(SUBJECT) + set[j] + str(classes[i]) + '.fdt'
            fid = open(path, 'rb')
            data = fromfile(fid, dtype('f'))
            data = data.reshape((72, 1000, 22))
            X[j].append(transpose(data, (0,2,1)))

    return X


def make_basis(q, m, fi, fm):
    # Construção da base de senos e cossenos
    # q = Tamanho dos sinais na base (nt)
    # m = Número de senos e cossenos na base
    # fi = Frequencia inicial reduzida (fl/fs)
    # fm = Frequencia máxima reduzida (fh/fs)

    t = arange(q)

    X0 = zeros((q, 2*m))

    for i in range(m):
        
        # Calculamos a frequencia reduzida correspondente ao bin i
        f = fi + float(i) / m * (fm-fi)
        
        # Obtemos o seno e o cosseno dessa frequencia
        b_sin = sin(2*pi*f*t)
        b_cos = cos(2*pi*f*t)
        
        # Colocamos os dois nas posições 2*i e 2*i+1 da base, de forma que ela fique organizada da seguinte forma
        # [Seno_0, Cosseno_0, Seno_1, Cosseno_1, etc]
        X0[:, 2*i] = b_sin
        X0[:, 2*i+1] = b_cos
    return X0


def filtering(X, fl, fh, m, fs=250.):
    # Filtragem usando projeção em base de senos e cossenos.
    # X = tensor de dados, organizado em um tensor de dimensão (ne, nc, nt):
    # ne = Número de épocas (144 no total, 72 para cada classe)
    # nc = Número de canais (22)
    # nt = Número de amostras no tempo (500, 2 segundos em 250 Hz)
    # fs = Frequencia de amostragem
    # fl = Frequencia mínima

    ne = X.shape[0]
    nt = X.shape[2]
    
    # Com a função make_basis criamos uma base de senos e cossenos, de dimensão (nt, 2*m)
    B = make_basis(nt, m, fl/fs, fh/fs)
    
    # G0 é a matriz de projeção, que liga diretamente o sinal original e o sinal projetado na base de interesse
    G0 = dot(pinv(dot(B.T, B)), B.T)

    # Para cada época do vetor X fazemos esta projeção, multiplicando a matriz X[k,:,:] por G0
    XF = asarray([dot(X[k,:,:], G0.T) for k in range(ne)])
    
    return XF


def get_step(n_bins, n_bands, overlap):
    # Função usada para obter o step (intervalo de coeficientes entre duas bandas)
    # Estimativa inicial do step, que seria o número de bins sobre o número de bandas truncado em múltiplos de 2
    step = (n_bins / n_bands / 2) * 2
    size = step * overlap

    # Porém as vezes o resultado retornado faz com que a última banda tenha bins não existentes.
    # Para corrigir isso usei um loop que verifica o último bin da última banda e se ele não for menor ou igual ao
    # número de bins o step é reduzido por 2. Provavelmente existe alguma maneira de achar isso de forma não-iterativa
    # mas como o tempo em que isso é feito é desprezível não procurei otimizar essa parte
    while True:
        last_end = (n_bands-1) * step + size
        if last_end <= n_bins:
            break
        step -= 2
        size = step * overlap

    return step


def windowing(X, t_0, t_start, t_end, fs):
    W_Start = int((t_start - t_0) * fs)
    W_End = int((t_end - t_0) * fs)
    for i in range(2):
        for j in range(2):
            X[i][j] = X[i][j][:, :, W_Start:W_End]
    return X


class CSP():

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):

        e, c, t = X.shape
        classes = unique(y)
        
        X0 = X[classes[0] == y,:,:]
        X1 = X[classes[1] == y,:,:]

        # Sum up covariance matrix
        S0 = zeros((c, c))
        S1 = zeros((c, c))
        for i in range(e / 2):

            S0 += dot(X0[i,:,:], X0[i,:,:].T)
            S1 += dot(X1[i,:,:], X1[i,:,:].T)

        [D, W] = eigh(S0, S0 + S1)

        ind = empty(c, dtype=int)
        ind[0::2] = arange(c - 1, c // 2 - 1, -1)
        ind[1::2] = arange(0, c // 2)
        
        W = W[:, ind]
        
        self.filters_ = W.T[:self.n_components]

    def transform(self, X):
    
        XT = asarray([dot(self.filters_, epoch) for epoch in X])
        XVAR = log(mean(XT ** 2, axis=2))
        
        return XVAR


def sbcsp(subject, classes, args):

    fl, fh, m, n_components, Clog, n_bands = args
    m = int(m)
    n_bands = int(n_bands)

    # Carregando os dados da memória. X contém dados de treinamento e validação de duas classes
    X = load_data(subject, classes)
    # print('Raw = ', len(X[0]), len(X[0][0]), len(X[0][0][0]), len(X[0][0][0][0]))

    # Todos os dados são janelados entre 2.5 e 4.5 segundos.
    # Fornecendo também o tempo inicial nesses arquivos que é igual a 2 segundos e a frequência de amostragem, 250 Hz
    X = windowing(X, 2.5, 2.5, 4.5, 250)
    # print('Windowed = ', len(X[0]), len(X[0][0]), len(X[0][0][0]), len(X[0][0][0][0]))

    # Aqui eu junto os dados de treinamento e validação cada um em um tensor (XT e XV respectivamente).
    # Cada um tem as dimensões (ne, nc, nt), e estão organizados de forma que as épocas correspondentes à primeira
    # classe venham primeiro e depois as da segunda classe
    # ne = Número de épocas (144 no total, 72 para cada classe)
    # nc = Número de canais (22)
    # nt = Número de amostras no tempo (500, 2 segundos em 250 Hz)
    XT = concatenate(X[0])
    XV = concatenate(X[1])
    # print('XT concatened(T+E) = ', XT.shape)
    # print('XV concatened(T+E) = ', XV.shape)
    # pyplot.plot(XT[1,:,:])
    # pyplot.show()

    # Definindo um vetor de tamanho 144, onde a primeira metade são todos 0 e a segunda metade são todos 1.
    # Isso vai ser usado como vetor de classes (gabarito)
    y = concatenate([zeros(72), ones(72)])
    # print('Y = ', y.shape)

    # Aqui é feita a projeção dos sinais na base de senos e cossenos.
    # São fornecidos as frequências limite (fl, fh) e o número de senos/cossenos na base (m).
    # Os tensores na saída são da forma (ne, nc, 2*m) e os coeficientes de seno/cosseno ficam intercalados de forma que
    # o vetor fique ordenado de acordo com a frequencia. Detalhes de como isso é feito estão na função filtering
    XT = filtering(XT, fl=fl, fh=fh, m=m)
    XV = filtering(XV, fl=fl, fh=fh, m=m)
    # print('XT Filtered = ', XT.shape)

    # Aqui são definidas algumas variaveis usadas para a divisão de bandas
    # n_bins = Número de bins de frequencia (2*m)
    # overlap = Overlap entre as bandas, nesse caso é de 50% (1/overlap)
    n_bins = 2*m
    overlap = 2
    
    # step = Intervalo de coeficientes entre duas bandas, resultado do número de bins, bandas e do overlap definido.
    # Mais detalhes na função get_step
    step = get_step(n_bins, n_bands, overlap)
    
    # size = Número de coeficientes em cada banda
    size = step * overlap

    # Aqui as bandas são separadas. Cada banda indexada por i é um pedaço do tensor original de dimensão (ne, nc, 2*m).
    # Após isso são obtidos n_bands bandas, cada banda sendo representada por um tensor de dimensão (ne, nc, size)
    XT = [XT[:, :, i*step:i*step+size] for i in range(n_bands)]
    XV = [XV[:, :, i*step:i*step+size] for i in range(n_bands)]

    # Vários (n_bands) CSPs são definidos e treinados com os dados do conjunto de treinamento (XT).
    # XT[i] é o tensor pra banda i, e o vetor y de classes é o mesmo pra todos
    csp = [CSP(n_components=n_components) for i in range(n_bands)]
    for i in range(n_bands):
        csp[i].fit(XT[i], y)

    # Depois do treinamento os dados são transformados, cada XT[i] e XV[i] vira uma matriz de forma (ne, n_components).
    # Mais detalhes do treinamento e transformação estão na classe CSP
    XT_CSP = [csp[i].transform(XT[i]) for i in range(n_bands)]
    XV_CSP = [csp[i].transform(XV[i]) for i in range(n_bands)]

    # Definindo uma matriz de dimensão (ne, n_bands)
    SCORE_T = zeros((144,n_bands))
    SCORE_V = zeros((144,n_bands))

    # Da mesma forma que o CSP, um LDA é treinado para cada banda
    clf = [LinearDiscriminantAnalysis() for i in range(n_bands)]
    for i in range(n_bands):
        clf[i].fit(XT_CSP[i], y)

        # Para cada banda, esse LDA treinado transforma os dados de treinamento e validação.
        # Relembrando, não é feita uma classificação 'dura', o resultado contínuo do LDA é mantido para construir um
        # vetor de características de dimensão (n_bands)
        SCORE_T[:,i] = ravel(clf[i].transform(XT_CSP[i]))
        SCORE_V[:,i] = ravel(clf[i].transform(XV_CSP[i]))

    # Aqui é treinado o meta-classificador bayesiano.
    # Os dados de treinamento são separados entre os pertencentes à classe 0 e classe 1. Para cada classe, calculamos
    # a média e o desvio padrão entre as épocas, de forma que os vetores sejam de dimensão n_bands
    SCORE_T0 = SCORE_T[y==0,:]
    m0 = mean(SCORE_T0, axis=0)
    std0 = std(SCORE_T0, axis=0)
    
    SCORE_T1 = SCORE_T[y==1,:]
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

    # A partir dos meta-scores um SVM linear é treinado e aplicado nos dados de validação
    svc = SVC(kernel="linear", C=10**Clog)
    svc.fit(META_SCORE_T, y)
    ans = svc.predict(META_SCORE_V)

    # ans == y retorna 1 quando a classe estimada é correta (igual a y) e 0 quando é incorreta.
    # Então a média desse vetor é igual a taxa de acertos do classificador
    return mean(ans == y)


if __name__ == "__main__":

    fl = 0
    fh = 51
    m = 100
    n_components = 6
    Clog = -4
    n_bands = 33
    
    args = (fl, fh, m, n_components, Clog, n_bands)

    subject = 1
    classes = [3, 4]
    acc_test = sbcsp(subject, classes, args)
    print(str(subject) + str(classes) + ' accuracy:' + str(acc_test*100))
    '''
    subjectsT = arange(1, 10)  # arange(1,10)
    classesT = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    att = zeros((len(subjectsT), len(classesT)))
    attime = zeros((len(subjectsT), len(classesT)))
    i = 0
    j = 0
    fullt = 0
    for SUBJECT, i in zip(subjectsT, range(10)):  # range(10)
        for classes, j in zip(classesT, range(7)):  # range(7)
            # print(su, classesT[j])
            t0 = time()
            try:
                acc_test = sbcsp(SUBJECT, classes, args)
            except (KeyboardInterrupt, SystemExit):
                raise
            t1 = time()
            print ('--------------------------------')
            print(i + 1, classesT[j], ' Time:', round(t1 - t0, 2), ' Acc:', round(acc_test * 100, 2))
            att[i, j] = acc_test
            attime[i, j] = t1 - t0
            fullt += t1 - t0
            stdout.flush()
    print('\nMean test accuracy: ' + str(round(mean(att) * 100, 2)))
    print('\nMean test time: ' + str(round(mean(attime), 2)))
    print('\nFull time: ' + str(round(fullt, 2)))
    '''