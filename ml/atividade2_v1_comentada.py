# -*- coding: utf-8 -*-
# =============================================================================
# Considere o mesmo conjunto de treinamento {x1,t1, · · · , xN,tN} da Atividade 01.
# 1. Considere o parâmetro de regularização λ e gere as curvas da Figura 1.7;
# 2. Apresente os valores de w, equivalentes aos da Tabela 1.2;
# 3. Apresente os resultados da Figura 1.8;
# 4. Com base na Seção 1.2.5, discuta os resultados obtidos a partir da regularização e gere a Figura 1.16.
# 5. Com base na Figura 1.21 e na Eq. 1.74, discuta a ”Maldição da Dimensionalidade”.
# =============================================================================
# regularização controla a magnitude e o overfitting
# qto maior o lambda menor a esfera de contenção de w. lambda sempre positivo e em geral entre 0 e 1. ln lambda = ⁻18 -> e^-18

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats
from scipy.stats import norm

# F(x): Função alvo - Para fins didaticos é conhecida, mas normalmente é desconhecida! ´E o que deseja-se aprender.
plt.figure()
x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$") #label="t = sen(r'$2\pix$)"

## Dataset Treinamento com rótulos=alvo=target -> mapeamento de x para y (x --> y)
x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.legend()

## Dataset Teste (N maior - mais dados)
x_test = np.linspace(0,1,1000, endpoint=False)
noise = np.random.randn(len(x_test))
# noise = np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
t_test = np.sin(2*np.pi*x_test) + 0.1 * noise # esse mesmo processo gerou a amostra de treinamento - aki, somente para ilustração - em geral não se conhece os alvos em dados de teste

#plt.figure()
#plt.scatter(x_test, t_test, facecolor="none", edgecolor="b", s=50, label="training data")
#plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
#plt.legend()


# Q1.
## Modelo de aprendizagem - Função Hipótese (g(x)) que quer aproximar f(x)
M = 9# complexidade do modelo - n° de parametros livres - que na forma polinomial define o grau progressivo do polinomio
# ... para representar as M+1 características dos dados de entrada

# ------------------------------------------------------
# previsões sob dados treinamento SEM regularização (idem atividade 1)
A_train = np.asarray([x_train**i for i in range(M+1)]).T # modelo polinomial - não linear - a transformação de x provocada pela potência progressiva regida por M torna o modelo não linear em x
# A é M+1 para considerar já com o bias = x0 = 1 representando o limiar que foi convertido para peso w0
# matriz A = [ [x⁰],[x¹],[x²],...,[x^M] ].T
w = np.linalg.inv(((A_train.T).dot(A_train))).dot((A_train.T).dot(t_train)) # encontrar vetor de pesos para cada característica de X que comporão a função hipótese polinomial g(x) - w é linear - diferentemente em w onde não há transformação polinomial
## Aplicando modelo aos próprios dados que o treinaram
y_train = A_train.dot(w) # y é a saida do modelo = f(x,w) = A*w
#plt.figure()
#plt.title("Previsões sob dados treinamento SEM regularização")
#plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
#plt.scatter(x_train, y_train, facecolor="none", edgecolor="r", s=50, label="predicted train $y(x,w)$")
##plt.plot(x_train, y_train, c="r", label="predicted test $y(x,w)$")
#plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
#plt.legend()

# previsões sob dados teste SEM regularização
A_test = np.asarray([x_test**i for i in range(M+1)]).T
y_test = np.dot(A_test,w)
plt.figure()
plt.title("Previsões sob dados TESTE sem regularização")
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
#plt.scatter(x_test, y_test, facecolor="none", edgecolor="r", s=50, label="predicted test $y(x,w)$")
plt.plot(x_test, y_test, c="r", label="predicted test $y(x,w)$")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()
plt.ylim(-2,2)

# ------------------------------------------------------
# Previsões sob dados treinamento com regularização ln lambda = -18
w = np.linalg.inv( (A_train.T).dot(A_train) + (np.exp(-18) * np.identity(M+1)) ).dot( (A_train.T).dot(t_train) ) # gerando w a partir de uma regularização ln lambda = -18
y_train = np.dot(A_train,w)
plt.figure()
plt.title("Previsões sob dados treinamento com regularização $\ln{\lambda}=-18$")
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
plt.scatter(x_train, y_train, facecolor="none", edgecolor="r", s=50, label="predicted train $y(x,w)$")
#plt.plot(x_train, y_train, c="r", label="predicted test $y(x,w)$")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()

# Previsões sob dados teste com regularização ln lambda = -18
A_test = np.asarray([x_test**i for i in range(M+1)]).T
y_test = np.dot(A_test,w)
plt.figure()
plt.title("Previsões sob dados TESTE com regularização $\ln{\lambda}=0$")
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
#plt.scatter(x_test, y_test, facecolor="none", edgecolor="r", s=50, label="predicted test $y(x,w)$")
plt.plot(x_test, y_test, c="r", label="predicted test $y(x,w)$")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()
plt.ylim(-2,2)

# ------------------------------------------------------
# Previsões sob dados treinamento com regularização ln lambda = 0
w = np.linalg.inv( (A_train.T).dot(A_train) + (np.exp(0) * np.identity(M+1)) ).dot( (A_train.T).dot(t_train) ) # gerando w a partir de uma regularização ln lambda = -18
y_train = np.dot(A_train,w)
plt.figure()
plt.title("Previsões sob dados treinamento com regularização $\ln{\lambda}=0$")
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
plt.scatter(x_train, y_train, facecolor="none", edgecolor="r", s=50, label="predicted train $y(x,w)$")
#plt.plot(x_train, y_train, c="r", label="predicted test $y(x,w)$")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()


# Previsões sob dados teste com regularização ln lambda = 0
A_test = np.asarray([x_test**i for i in range(M+1)]).T
y_test = np.dot(A_test,w)
plt.figure()
plt.title("Previsões sob dados TESTE com regularização $\ln{\lambda}=0$")
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
#plt.scatter(x_test, y_test, facecolor="none", edgecolor="r", s=50, label="predicted test $y(x,w)$")
plt.plot(x_test, y_test, c="r", label="predicted test $y(x,w)$")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()
plt.ylim(-2,2)

   

# Q2.
M=9
lambd = [-np.inf,-18,0]
W = np.zeros([M+1,len(lambd)]) 
for l,k in zip(lambd,range(len(lambd))):
    A = np.asarray([x**i for i in range(M+1)]).T # gerando a matriz A = [ [x⁰],[x¹],[x²],...,[x^M] ].T
    w = np.linalg.inv( (A.T).dot(A) + (np.exp(l) * np.identity(M+1)) ).dot( (A.T).dot(t) )
    for j in range(len(w)): W[j,k] = w[j]

# Q3.
M=9
lambd = np.arange(-40,-20)
Erms = []
ErmsV = []
xv = np.linspace(0.1,0.8,100, endpoint=False)
tv = np.sin(2*np.pi*xv) + 0.1*np.random.randn(len(xv))
for l in lambd:
    A = np.asarray([x**i for i in range(M+1)]).T # gerando a matriz A = [ [x⁰],[x¹],[x²],...,[x^M] ].T
    w = np.linalg.inv( (A.T).dot(A) + (np.exp(l) * np.identity(M+1)) ).dot( (A.T).dot(t) )
    y = np.dot(A,w)
    E = ((y-t).T).dot(y-t) # (np.dot((np.dot(A,w)-t).T,(np.dot(A,w)-t)))/2
    Erms.append(sqrt((E)/len(x)))
    
    Av = np.asarray([xv**j for j in range(M+1)]).T
    yv = np.dot(Av,w)
    Ev = ((yv-tv).T).dot(yv-tv)
    ErmsV.append(sqrt(Ev/len(xv)))
    
plt.figure(figsize=(10,7))
plt.plot(lambd,Erms,c='b', marker='o', label='Training')
plt.plot(lambd,ErmsV,c='r', marker='o', label='Test')
plt.legend(loc="best")
plt.yticks(np.linspace(0,2,5, endpoint=True))
plt.xticks([-40,-35,-30,-25,-20])
plt.ylim(0,2)
    
# Q4. Incompleta, não entendi a incorporação da gaussiana. quem é x0
M=9
A = np.asarray([x**i for i in range(M+1)]).T # gerando a matriz A = [ [x⁰],[x¹],[x²],...,[x^M] ].T
w = np.linalg.inv( (A.T).dot(A) + (np.exp(-18) * np.identity(M+1)) ).dot( (A.T).dot(t) )
y = np.dot(A,w)
#plt.plot(x,y,c='k')

xv = np.linspace(0.1,1,1000, endpoint=False)
Av = np.asarray([xv**i for i in range(M+1)]).T
yv = np.dot(Av,w)
plt.plot(xv,yv,c='r')
plt.scatter(Av[0,:],yv[0])
plt.ylim(-2,2)

a = np.arange(-2,yv[-1])
b = a*0+Av[0,:]
plt.plot(b,a,c='k')

# np.var(x)
#plt.plot(xv,norm.pdf(yv, yv.mean(), yv.std()))


x = [8,2,10,10]
x = np.linspace(0,1,1000)
ruido = np.random.normal(scale=0.25, size=len(x)) # np.random.randn(len(x)) 
x = np.sin(2*np.pi*x) + 0.1*ruido

# Variância e desvio padrão são medidas de variabilidade e medem a dispersão (dos 
# dados em relação à média
# variancia = soma dos quadrados das diferenças entre cada observação e a média 
# das observações dividido pelo número de observações
var1 = np.sum(np.asarray(x-np.mean(x))**2)/len(x) 
var2 = np.var(x)

# desvio padrão = 
std1 = math.sqrt(var1)
std2 = np.std(x)

normal1 = ( 1/math.sqrt(2*np.pi*var1) ) * np.exp( (-1/(2*var1))*((x-np.mean(x))**2)  )

normal2 = norm.pdf(x, np.mean(x), np.std(x))

# ============================Distribuição Normal=================================================
# # Conjunto de objetos em uma cesta, a média é 8 e o desvio padrão é 2
# # Qual a probabilidade de tirar um objeto que peso é menor que 6 quilos?
# norm.cdf(6, 8, 2)
# # Qual a probabilidade de tirar um objeto que o peso á maior que 6 quilos?
# norm.sf(6, 8, 2)
# 1 - norm.cdf(6, 8, 2)
# # Qual a probabilidade de tirar um objeto que o peso é menor que 6 ou maior que 10 quilos?
# norm.cdf(6, 8, 2) + norm.sf(10, 8, 2)
# # Qual a probabilidade de tirar um objeto que o peso é menor que 10 e maior que 8 quilos?
# norm.cdf(10, 8, 2) - norm.cdf(8, 8, 2)
# 
# dados = norm.rvs(size = 100) # GERA DADOS EM UMA DISTRIBUIÇÃO NORMAL
# stats.probplot(dados, plot = plt)
# stats.shapiro(dados)
# =============================================================================
