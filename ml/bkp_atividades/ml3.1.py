# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct)

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

x_test = np.linspace(0.01,0.99,100, endpoint=False)
noise = np.random.randn(len(x_test))
# noise = np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
t_test = np.sin(2*np.pi*x_test) + 0.1 * noise 

##### Questão 1.3 ############
# Para um vetor de Ms
M = [0,1,3,9] 

var = np.var(x_train) #s - tem interpretação de variancia (pode ser calculada a partir dos blocos de x_train que calculam cada mu)

x_train1 = x_train[:, None]
x_test1 = x_test[:, None]
  
plt.subplots(1, figsize=(10,6))
for j, grau in enumerate(M):

    mu = np.linspace(x_train[0], x_train[-1], grau)
    
    mu = mu[:, None]
       
    #assert np.size(x_train1, 1) == np.size(mu, 1)
    base = [np.ones(len(x_train1))]
    for m in mu: base.append(np.exp(-0.5 * np.sum(np.square(x_train1 - m), axis=-1) / var))
    A_train = np.asarray(base).T
     
    #assert np.size(x_test1, 1) == np.size(mu, 1)
    base = [np.ones(len(x_test1))]
    for m in mu: base.append(np.exp(-0.5 * np.sum(np.square(x_test1 - m), axis=-1) / var))
    A_test = np.asarray(base).T
      
#    A_train = np.zeros([len(x_train),grau+1])
#    A_train[:,0] = np.ones([len(x_train)])
#    for i in range(grau): A_train[:,i+1] = np.asarray([(np.exp((-1)*(((xi - mu[i])**2)/2*(s**2)))).T for xi in x_train]).T
#    
#    A_test = np.zeros([len(x_test),grau+1])
#    A_test[:,0] = np.ones([len(x_test)])
#    for i in range(grau): A_test[:,i+1] = np.asarray([(np.exp((-1)*(((xi - mu[i])**2)/2*(s**2)))).T for xi in x_test]).T
    
    w = np.linalg.pinv(A_train) @ t_train # inv(A.T * A) * A.T * t 
    
    y_train = A_train @ w 
    
    y_test = A_test @ w 

    plt.subplot(2,2,j+1)
    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
    plt.plot(x_correct, t_correct, c="g", alpha=0.6, label="$\sin(2\pi x)$")
    plt.plot(x_test, y_test, c="r", label="$y(x,w)$") #fitting
    plt.ylabel('$t$', fontsize=13, rotation = 360)
    plt.xlabel('$\mathit{x}$', fontsize=13,labelpad=-15)
    plt.xticks([0,1])
    plt.yticks([-1,0,1])
    plt.ylim(-1.5,1.5)
    plt.annotate(('M={}'.format(grau)), xy=(0.8, 1))
    
plt.legend(bbox_to_anchor=(1.05, 2.2), loc=2, borderaxespad=0.)
#plt.legend(loc='lower left')
plt.savefig('figures/Fig3.1.3.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


######## Questão 1.4 ############
M = range(10)
var = np.var(x_train)
Erms = []
ErmsV = []
# Cálculo do erro de previsão em relação ao alvo - Dados de treinamento - Busca-se minimizar esse erro
# Calculo do erro a partir de modelos com várias complexidades M=[0 até 9]
for i in M:
    mu = np.linspace(x_train[0], x_train[-1], i)
    mu = mu[:, None]
       
    base = [np.ones(len(x_train1))]
    for m in mu: base.append(np.exp(-0.5 * np.sum(np.square(x_train1 - m), axis=-1) / var))
    A_train = np.asarray(base).T
     
    base = [np.ones(len(x_test1))]
    for m in mu: base.append(np.exp(-0.5 * np.sum(np.square(x_test1 - m), axis=-1) / var))
    A_test = np.asarray(base).T
    
    w = np.linalg.pinv(A_train) @ t_train # inv(A.T * A) * A.T * t 
    y_train = A_train @ w 
    y_test = A_test @ w
    
    Erms.append(  np.sqrt( np.mean( np.square( y_train - t_train )))) # bias decomposition ~ math.sqrt( (y_train-t_train).T.dot(y_train-t_train)/len(x_train) )
    ErmsV.append( np.sqrt( np.mean( np.square( y_test  - t_test  )))) # variance decomposition ~ math.sqrt( (y_test - t_test).T.dot(y_test - t_test)/len(x_test) )

plt.figure(figsize=(10,7))                 
plt.plot(Erms, 'o-', mfc="none", mec="b", ms=10, c="b", label="Training")
plt.plot(ErmsV, 'o-', mfc="none", mec="r", ms=10, c="r", label="Test")
plt.legend(loc="best")
plt.xlabel('M', fontsize=16)
plt.ylabel('$E_{RMS}$', fontsize=16)
plt.xticks(np.linspace(0,9,10))

# plt.yticks(np.linspace(0,1,6)) # retire o comentário para zoom na escala
# plt.ylim(0,1) # retire o comentário para zoom na escala

plt.savefig('figures/Fig3.1.4.1.png', format='png', dpi=300, transparent=True, bbox_inches='tight')


######## Questão 1.5 ############
M = [0,1,3,9]

var = np.var(x_train)

W = np.zeros([M[-1]+1,len(M)]) 

for m,k in zip(M,range(len(M))):
    
    mu = np.linspace(x_train[0], x_train[-1], m)
    mu = mu[:, None]
       
    base = [np.ones(len(x_train1))]
    for m in mu: base.append(np.exp(-0.5 * np.sum(np.square(x_train1 - m), axis=-1) / var))
    A_train = np.asarray(base).T
    
    w = np.linalg.pinv(A_train) @ t_train # inv(A.T * A) * A.T * t 
    
    for j in range(len(w)): W[j,k] = w[j]




# =============================================================================
# ######### Para M único ############
# M = 4
# 
# ### 1) separar x em 3 e obter a média de x para cada região..
# #mu = np.array([np.mean(x[0:3]), np.mean(x[3:6]), np.mean(x[6:])]) 
# 
# ### 2) gerar números aleatórios dentro da faixa de valores de x
# #pos = np.random.randint(len(x_train)-1, size=M)
# #mu = np.sort([x_train[i] for i in pos]) 
# 
# ### 3) adotar a qtd de mu igual à N, assim haverá uma gaussiana vcentrada em cada elemento de x (para determinar os centros pode-se utilizar o algoritmo o k-means)   
# #interv = int(len(x_train)/M) # calculando intervalo de indices entre os pontos médios x_train 
# #pos = [ interv*i for i in range(1,M+1) ] # definindo os indices dos pontos médios de x_train a partir dos intervalos
# #mu = x_train[pos] # capturando os valores dos pontos médios de x_train para a gaussiana
# mu = np.linspace(x_train[0], x_train[-1], M) # oooouuuuuu usar o linspace para tudo isso :]
# 
# # faixa = [ x_train[pos[i]:pos[i+1]] for i in range(0,len(pos)) ] # calculo da variância de cada intervalo
# var = np.var(x_train) #s - tem interpretação de variancia (pode ser calculada a partir dos blocos de x_train que calculam cada mu)
# 
# x_train1 = x_train[:, None]
# mu = mu[:, None]      
# #assert np.size(x_train1, 1) == np.size(mu, 1)
# base = [np.ones(len(x_train1))]
# for m in mu: base.append(np.exp(-0.5 * np.sum(np.square(x_train1 - m), axis=-1) / var))
# A_train = np.asarray(base).T
# 
# x_test1 = x_test[:, None]     
# #assert np.size(x_test1, 1) == np.size(mu, 1)
# base = [np.ones(len(x_test1))]
# for m in mu: base.append(np.exp(-0.5 * np.sum(np.square(x_test1 - m), axis=-1) / var))
# A_test = np.asarray(base).T
# 
# w = np.linalg.pinv(A_train) @ t_train
# 
# y_train = A_train @ w
#    
# Err_train = np.mean(np.square(y_train - t_train)) # bias decomposition
# 
# y_test = A_test @ w
# 
# Err_test = np.mean(np.square(y_train - t_train)) 
# 
# y_test_rmse = np.sqrt(Err_train) + np.zeros_like(y_test) # variance decomposition
# 
# plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
# plt.plot(x_correct, t_correct, label="$\sin(2\pi x)$")
# plt.plot(x_test, y_test, label="prediction")
# plt.fill_between( 
#         x_correct, y_test - y_test_rmse, y_test + y_test_rmse,
#         color="orange", alpha=0.5, label="std.")
# plt.legend()
# plt.ylim(-1.5,1.5)
# =============================================================================
