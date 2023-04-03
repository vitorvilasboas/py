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

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

x_test = np.linspace(0,1,1000, endpoint=False) #0.1-0.8
t_test = np.sin(2*np.pi*x_test) + 0.1 * np.random.randn(len(x_test))
#t_test = np.sin(2*np.pi*x_test) + 0.1 * np.random.normal(scale=0.25, size=x_test.shape) #scale=desvio padrão

M = 9
A_train = np.asarray([x_train**i for i in range(M+1)]).T
A_test = np.asarray([x_test**i for i in range(M+1)]).T

# Q1.
w = np.linalg.inv(((A_train.T).dot(A_train))).dot((A_train.T).dot(t_train))
y_train = A_train.dot(w)
y_test = np.dot(A_test,w)
plt.figure()
plt.title("Previsões sob dados TESTE sem regularização")
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
plt.plot(x_test, y_test, c="r", label="predicted test $y(x,w)$")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()
plt.ylim(-2,2)

# w1 = np.linalg.solve(np.exp(-18) * np.eye(np.size(A_train, 1)) + A_train.T @ A_train, A_train.T @ t_train) #1e-18
w = np.linalg.inv( (A_train.T).dot(A_train) + (np.exp(-18) * np.identity(M+1)) ).dot( (A_train.T).dot(t_train) ) 
y_train = np.dot(A_train,w)
y_test = np.dot(A_test,w)
plt.figure()
plt.title("Previsões sob dados TESTE com regularização $\ln{\lambda}=0$")
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
plt.plot(x_test, y_test, c="r", label="predicted test $y(x,w)$")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()
plt.ylim(-2,2)

w = np.linalg.inv( (A_train.T).dot(A_train) + (np.exp(0) * np.identity(M+1)) ).dot( (A_train.T).dot(t_train) ) 
y_train = np.dot(A_train,w)
y_test = np.dot(A_test,w)
plt.figure()
plt.title("Previsões sob dados TESTE com regularização $\ln{\lambda}=0$")
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
plt.plot(x_test, y_test, c="r", label="predicted test $y(x,w)$")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.legend()
plt.ylim(-2,2)

   
# Q2.
lambd = [-np.inf,-18,0]
W = np.zeros([M+1,len(lambd)]) 
for l,k in zip(lambd,range(len(lambd))):
    w = np.linalg.inv( (A_train.T).dot(A_train) + (np.exp(l) * np.identity(M+1)) ).dot( (A_train.T).dot(t_train) )
    for j in range(len(w)): W[j,k] = w[j]

# Q3.
x_test = np.linspace(0,1,1000, endpoint=False) #0.1-0.8
t_test = np.sin(2*np.pi*x_test) + 0.1 * np.random.randn(len(x_test))
A_test = np.asarray([x_test**i for i in range(M+1)]).T

lambd = np.arange(-40,-3)
Erms = []
ErmsV = []
for l in lambd:
    # quanto maior np.exp(l), maior o lâmbda e maior o impacto sobre o erro
    # a medida que lâmbda cresce e tende a 0 o impacto sobre o erro diminui
    w = np.linalg.inv( (A_train.T).dot(A_train) + (np.exp(l) * np.identity(M+1)) ).dot( (A_train.T).dot(t_train) )
    y_train = np.dot(A_train,w)
    E = ((y_train-t_train).T).dot(y_train-t_train) + ((np.exp(l)/2)*(w.T).dot(w))
    Erms.append(sqrt((E)/len(x_train)))
    
    y_test = np.dot(A_test,w)
    Ev = ((y_test-t_test).T).dot(y_test-t_test) + ((np.exp(l)/2)*(w.T).dot(w))
    ErmsV.append(sqrt((Ev)/len(x_test)))
    
plt.figure(figsize=(10,7))
plt.plot(lambd,Erms,c='b', label='Training')
plt.plot(lambd,ErmsV,c='r', label='Test')
plt.legend(loc="best")
#plt.yticks(np.linspace(0,2,5, endpoint=True))
#plt.xticks([-40,-35,-30,-25,-20])
plt.ylim(0,10)


# Q4. 
w = np.linalg.inv(A_train.T.dot(A_train) + np.exp(-18) * np.identity(M+1)).dot(A_train.T.dot(t_train))
y_train = np.dot(A_train,w)
z = (y_train-t_train).T.dot(y_train-t_train) + np.exp(-18)/2 * w.T.dot(w)
zr = sqrt(E/len(x_train))


# expressar nossa incerteza sobre o valor da variável alvo usando uma distribuição de probabilidade. 
# Para esse propósito, assumiremos que, dado o valor de x, o valor correspondente de t possui uma distribuição 
# gaussiana com uma média igual ao valor y (x, w) da curva polinomial - bet
# Modelo de Regressão Bayesiano
# w ~ N(w|0, lambd^(-1) * I)
# y = A @ w
# t ~ N(t|A @ w, beta^(-1))

lambd = np.exp(-18) # 2e-3
beta = 2 # parâmetro de precisão beta

ndim = np.size(A_train, 1)
mean_prev, precision_prev = np.zeros(ndim), lambd * np.eye(ndim)
w_precision = precision_prev + beta * A_train.T @ A_train
w_mean = np.linalg.solve( w_precision, precision_prev @ mean_prev + beta * A_train.T @ t_train )
w_cov = np.linalg.inv(w_precision)

# Calculando a média e o desvio padrão da distribuição preditiva (dp)
# w_sample = np.random.multivariate_normal(w_mean, w_cov, size=sample_size ) # if sample_size is not None: - número de amostras a serem retiradas da distribuição preditiva
# y_sample = A_test @ w_sample.T   # amostras da dp
y = A_test @ w_mean # média da dp
y_var = 1 / beta + np.sum(A_test @ w_cov * A_test, axis=1) 
y_err = np.sqrt(y_var) # y_std = y_err  # desvio padrao da dp

plt.figure(figsize=(8,5))
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y, c="r", label="mean")
plt.fill_between(x_test, y - y_err, y + y_err, color="pink", label="std.", alpha=0.5)
plt.xlim(-0.1, 1.1)
plt.ylim(-1.5, 1.5)
plt.annotate("M=9", xy=(0.8, 1))
plt.legend(bbox_to_anchor=(1.05, 1.), loc='best', borderaxespad=0.)


#x_test = np.arange(-10, 10, 0.001)
#y2 = norm.pdf(x_test,y,y_err)
y2 = norm.pdf(w_mean,y,y_err)
plt.plot(y2+0.5, x_test)
#plt.fill_between(y2,x_test,0, alpha=0.3, color='b')


#w = np.linalg.inv( (A_train.T).dot(A_train) + (np.exp(-18) * np.identity(M+1)) ).dot( (A_train.T).dot(t_train) ) 
#y_test = np.dot(A_test,w)
#plt.plot(x_test,y_test,c='r')
#plt.scatter(A_test[0,0],y_test[0])
##plt.ylim(-2,2)
#a = np.arange(-2,y_test[-1])
#b = a*0+A_test[0,:]
#plt.plot(b,a,c='k')

# np.var(x)
#plt.plot(xv,norm.pdf(yv, yv.mean(), yv.std()))


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
