import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(0,1,50)
T = np.sin(2*np.pi*X) + 0.1*np.random.randn(len(X))
N,D = 50,1# X.reshape(-1,1).shape
K = 1 #len(T)
M = 3
epocas = 1500
eta = 0.1
W1 = np.random.rand(M,D+1)
W2 = np.random.rand(K,M+1)
mse = []
for i in range(epocas):
    E = 0
    for n in range(N):
        x = X[n]
        t = T[n]     
        x = np.concatenate([np.ones(1),[x]])   
        aj = W1.dot(x) # soma ponderada   
        ha = 1./(1 + np.exp(-aj)) # função de ativação sigmoide
        ha = (np.exp(aj) - np.exp(-aj))/(np.exp(aj) + np.exp(-aj)) # func ativacao tangh
        zj = np.array(np.concatenate([np.ones(1), np.asarray(ha)]))
        y = W2.dot(zj)
        
        deltaK = y-t # erro * derivada de sigma(ak)
        dz = zj * (1-zj) # derivada de h(aj) sigmoid
        dz = 1-zj**2 # derivada de h(aj) tangh
        #deltaJ = np.array([ dz[i] * (W2.T @ deltaK)[i] for i in range(len(dz)) ]) # multiplicacao ponto a ponto
        deltaJ = dz * (W2.T @ deltaK)
        deltaJ = deltaJ[1:] # eliminando o bias para a retropropagacao
        zj = zj.reshape(-1,1)
        deltaK = deltaK.reshape(-1,1)
        x = x.reshape(-1,1)
        deltaJ = deltaJ.reshape(-1,1)
        W2 = W2 - eta * deltaK.dot(zj.T)
        W1 = W1 - eta * deltaJ.dot(x.T) 
        E += deltaK.T.dot(deltaK)
    mse.append(E/N)
mse = np.ravel(mse)
plt.plot(mse)