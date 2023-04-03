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

K = len(f_quad[0]) # num de saidas da rede

epocas = 1000 # n_iterações
eta = 0.001

fact = ['sigmoid','tanh']

colors = ['c','darkorange','lime','gray','r']
titles = [r'$x^2$', r'$\sin(x)$', r'$|x|$', r'$step(x)$']
markers = ['o','p','*','s','x']

plt.rcParams.update({'font.size': 12})

#plt.figure(figsize=(20, 10))
#plt.suptitle('Resultados para MLP de duas camadas ocultas e função de ativação {} para aproximação de 4 funções de base distintas, \n com 5 diferentes quantidades de nós nas camadas ocultas ($M$)'.format(h_a))
#M1 = np.random.randint(D+1,D+10) # num de nós na primeira camada oculta 
#M2 = np.random.randint(D+1,D+10) # num de nós na segunda camada oculta

#M = np.array(np.linspace(D+1, 40, 5), dtype=int) # sorted(np.unique(np.random.randint(D+1, D+50, 5)))
M = np.array([2, 10, 30, 100])
    
func_mse = []
for T, ftitle in zip(funcoes_alvo, range(1,5)):
    
    plt.figure(figsize=(20, 7))
    
    for h_a,lenf in zip(fact,range(1,3)):
        plt.subplot(1, 2, lenf)
        plt.title(r'$h(a)={}$'.format(h_a))
        plt.plot(X, T, color="g", lw=2, ls='-', label=titles[ftitle-1])
        plt.xticks([-1, -0.5, 0, 0.5, 1])
        plt.yticks([-1, 0, 1])
        #plt.ylim([-1.5, 1.1])
        #plt.xlabel('x', fontsize=13)
        #plt.ylabel(titles[ftitle-1], fontsize=13, rotation = 360, labelpad=13)
        # plt.title('Resultados para MLP de duas camadas ocultas e função de ativação {} para aproximação de 4 funções de base distintas, \n com 5 diferentes quantidades de nós nas camadas ocultas ($M$)'.format(h_a))
        
        for M1, cl, fig in zip(M, range(len(M)), range(1,5)):
            M2 = M1
        
            W1 = np.random.randn(M1,D+1)
            W2 = np.random.randn(M2,M1+1)
            W3 = np.random.randn(K,M2+1)
            
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
        
        plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1,-1), loc=2, borderaxespad=2)
    plt.savefig('Fig5.{}.png'.format(ftitle), format='png', dpi=300, transparent=True, bbox_inches='tight')