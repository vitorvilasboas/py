# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:38:06 2019
@author: Vitor Vilas-Boas
Machine Learning - Activity 5 - Neural Networks

5. Considere a Figura 5.3 do livro texto. Utilize uma rede neural do tipo perceptron 
multicamadas (MLP),  com duas camadas ocultas, para aproximar das seguintes funções: 
a) f(x) = x^2 
b) f(x) = sin(x)
c) f(x) = |x|
d) f(x) = step(x) = 1 | x ≥ 0; 0 | x < 0 
Assuma que x pode assumir N = 50 pontos aleatórios no intervalo |x| < 1.
5.1. Comente, com base na teoria, a influência do número de nós na camada oculta para 
a solução do problema de regressão. Apresente os resultados para, pelo menos, três valores distintos de M;
5.2. Assuma que as funções de ativação da rede neural são, uma por vez, a sigmoid e a 
tanh para a solução do problema. Comente os resultados obtidos. 
"""
import numpy as np
import matplotlib.pyplot as plt

N = 50 # num de exemplos nos dados de entrada da rede (x)
X  = np.linspace(-0.99, 1, N, endpoint=False)[:, None]
D = len(X[0]) # dimensão dos vetores de características de cada exemplo de x (componentes de cada xi E x)

f_quad = np.square(X) #x^2  
f_seno = np.sin(np.pi * X) #sin(x)
f_mod = np.abs(X) #|x|
f_step = np.array([1 if x >= 0 else 0 for x in X])[:, None] #step(x) = 1 | x ≥ 0; 0 | x < 0  <-> #0.5*(np.sign+1) 

funcoes_alvo = list([f_quad, f_seno, f_mod, f_step])

K = len(f_quad[0]) # num de sadas da rede

epocas = 1000 # n_iterações
eta = 0.001

#h_a = 'tanh' #sigmoid or tanh
fact = ['sigmoid','tanh']

colors = ['c','darkorange','lime','gray','r']
titles = [r'$x^2$', r'$\sin(x)$', r'$|x|$', r'$step(x)$']
markers = ['o','p','*','s','x']

#M1 = np.random.randint(D+1,D+10) # num de nós na primeira camada oculta 
#M2 = np.random.randint(D+1,D+10) # num de nós na segunda camada oculta

#M = np.array(np.linspace(D+1, 40, 5), dtype=int) # sorted(np.unique(np.random.randint(D+1, D+50, 5)))
M = np.array([2, 10, 30, 100])

for h_a,lenf in zip(fact,range(5,7)):
    
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(20, 14))
    #plt.suptitle('Resultados para MLP de duas camadas ocultas e função de ativação {} para aproximação de 4 funções de base distintas, \n com 5 diferentes quantidades de nós nas camadas ocultas ($M$)'.format(h_a))
    
    M_mse = []
    for M1, cl in zip(M, range(len(M))):
        M2 = M1
    
        W1 = np.random.randn(M1,D+1)
        W2 = np.random.randn(M2,M1+1)
        W3 = np.random.randn(K,M2+1)
        
        func_mse = []
        for fig, T in zip(range(1,5), funcoes_alvo):
            plt.subplot(2, 2, fig)
            
            mse = []
            for n_iter in range(epocas): 
            
                E = 0
                Y = []
                for n in range(N):
                            
                    ##### FORWARD ######
                    
                    # Entrada
                    x = X[n,:] # X[n,:].reshape(-1,1)
                    x = np.concatenate([np.ones(1),x]) #add bias <-> np.concatenate([np.ones((1,1)),xn], axis=1)
                    
                    # Primeira Camada Oculta
                    aj1 = W1.dot(x) # soma ponderada
                    if h_a == 'sigmoid': ha1 = 1./(1 + np.exp(-aj1)) # func ativação sigmoid
                    elif h_a == 'tanh': ha1 = (np.exp(aj1) - np.exp(-aj1))/(np.exp(aj1) + np.exp(-aj1)) # func ativação tanh
                    zj1 = np.array(np.concatenate([np.ones(1), np.asarray(ha1)]))
                    
                    # Segunda Camada Oculta
                    aj2 = W2.dot(zj1)
                    if h_a == 'sigmoid': ha2 = 1./(1 + np.exp(-aj2)) # func ativação sigmoid
                    elif h_a == 'tanh': ha2 = (np.exp(aj2) - np.exp(-aj2))/(np.exp(aj2) + np.exp(-aj2)) # func ativação tanh
                    zj2 = np.array(np.concatenate([np.ones(1), np.asarray(ha2)]))
                    
                    # Saída
                    y = W3.dot(zj2)
                    
                    ##### BACKWARD ######
                    
                    #### Calculo do Erro e Backpropagation
                    
                    t = T[n,:]
                    deltaK = y-t # erro * derivada de sigma(ak)
                    
                    # Impacto do erro sobre a segunda camada oculta
                    if h_a == 'sigmoid': dz2 = zj2 * (1-zj2)    # derivada de ha2 = h(aj2) sigmoid
                    elif h_a == 'tanh':  dz2 = 1 - zj2**2       # derivada de ha2 = h(aj2) tangh
                    #deltaJ2 = np.array([ dz2[i] * (W2.T @ deltaK)[i] for i in range(len(dz)) ]) # multiplicacao ponto a ponto
                    deltaJ2 = dz2 * (W3.T @ deltaK)
                    deltaJ2 = deltaJ2[1:] # eliminando o bias para a retropropagacao
                    
                    # Impacto do erro sobre a primeira camada oculta
                    if h_a == 'sigmoid': dz1 = zj1 * (1-zj1)    # derivada de ha1 = h(aj1) sigmoid
                    elif h_a == 'tanh':  dz1 = 1 - zj1**2       # derivada de ha1 = h(aj1) tangh
                    deltaJ1 = dz1 * (W2.T @ deltaJ2)
                    deltaJ1 = deltaJ1[1:] # eliminando o bias para a retropropagacao
                    
                    x = x.reshape(-1,1)
                    zj2 = zj2.reshape(-1,1)
                    zj1 = zj1.reshape(-1,1)
                    deltaK = deltaK.reshape(-1,1)
                    deltaJ2 = deltaJ2.reshape(-1,1)
                    deltaJ1 = deltaJ1.reshape(-1,1)
                    
                    # Atualizando pesos
                    W3 = W3 - eta * deltaK.dot(zj2.T)
                    W2 = W2 - eta * deltaJ2.dot(zj1.T)
                    W1 = W1 - eta * deltaJ1.dot(x.T) 
                    
                    # Soma dos erros entre y e t em cada exemplo x
                    E += deltaK.T.dot(deltaK)
                    
                    Y.append(y)
                Y = np.asarray(Y)
            
                mse.append(E/N) # Add Erro médio quadrado
            
            mse = np.ravel(mse)
            func_mse.append(mse)
            
            plt.plot(X, Y, color=colors[cl], label=r'$M={}$'.format(M[cl]), marker=markers[cl], markersize=3, ls='-', alpha=0.5)
            plt.xlabel('x', fontsize=13)
            plt.xticks([-1, -0.5, 0, 0.5, 1])
            plt.yticks([-1, 0, 1])
            plt.ylabel(titles[fig-1], fontsize=13, rotation = 360, labelpad=13)
            plt.plot(X, T, color="g", lw=2, ls='-')
        M_mse.append(func_mse)
    plt.legend(bbox_to_anchor=(1.05, 2.2), loc=2, borderaxespad=0.)  
    plt.savefig('Fig5.{}_{}.png'.format(lenf,h_a), format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    ######### M_MSE = mse M x func x iter
    
    MSE = np.transpose(np.asarray(M_mse), (1,0,2))
    
    # MSE_a = []
    # MSE_a.append(M_mse[0][1])
    # MSE_a.append(M_mse[1][1])
    # MSE_a.append(M_mse[2][1])
    # MSE_a.append(M_mse[3][1])
    
    # for fff in MSE_a: plt.plot(fff)

    plt.figure(figsize=(20, 14))
    #plt.suptitle('MSE para cada uma das {} iterações com função de ativação definida como {}'.format(epocas,h_a))        
    for fig, erro in zip(range(1,5), MSE):
        plt.subplot(2, 2, fig)
        plt.ylabel(r'$MSE$', fontsize=13, rotation = 360, labelpad=13)
        plt.xlabel('iteração', fontsize=13)
        plt.ylim(0,0.1)
        plt.annotate((r'$f(x)=${}'.format(titles[fig-1])), xy=(800, 0.07))
        [ plt.plot(erro[e], color=colors[e], label=r'$M={}$'.format(M[e]), ls='-') for e in range(0,4)]
        # for m, cont, e in zip(M, range(len(M)), erro): plt.plot(e, color=colors[cont], label=r'$M={}$'.format(M[cl]), ls='-')
      
    plt.legend(bbox_to_anchor=(1.05, 2.2), loc=2, borderaxespad=0.)  
    plt.savefig('Fig5.{}_{}_mse.png'.format(lenf+2,h_a), format='png', dpi=300, transparent=True, bbox_inches='tight')