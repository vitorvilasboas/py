import numpy as np
import matplotlib.pyplot as plt

# X = np.asarray([[0, 0, 1, 1],[0,1,0,1]])
X = np.array([[0,1,0,1],[0,0,1,1]])
T = np.array([[1,0,0,1],[0,0,0,1]])

M = 3
K = len(T)
epocas = 1500
eta = 0.1

D,N = X.shape

W1 = np.random.rand(M,D+1)
W2 = np.random.rand(K,M+1)

mse = []
for i in range(epocas):
    E = 0
    for n in range(N):
        x = X[:,n]
        t = T[:,n]
        
        x = np.concatenate([np.ones(1),x])
        
        aj = W1.dot(x) # soma ponderada
        
        ha = 1./(1 + np.exp(-aj)) # função de ativação
        
        zj = np.array(np.concatenate([np.ones(1), np.asarray(ha)]))
        
        y = W2.dot(zj)
        
        deltaK = y-t # erro * derivada de sigma(ak)
        
        dz = zj * (1-zj) # derivada de h(aj)
        
        #deltaJ = np.array([ dz[i] * (W2.T @ deltaK)[i] for i in range(len(dz)) ]) # multiplicacao ponto a ponto
        
        deltaJ = dz * (W2.T @ deltaK)
        
        deltaJ = deltaJ[1:]
        
        zj = zj.reshape(-1,1)
        deltaK = deltaK.reshape(-1,1)
        
        W2 = W2 - eta * deltaK.dot(zj.T)
        W1 = W1 - eta * deltaJ.dot(x.T) 
        
        E += deltaK.T.dot(deltaK)
    
    mse.append(E/N)
mse = np.ravel(mse)

msev = []   
Ev = 0
for n in range(N):
    x = X[:,n]
    t = T[:,n]
    
    x = np.concatenate([np.ones(1),x])
    
    aj = W1.dot(x) # soma ponderada
    
    ha = 1./(1 + np.exp(-aj)) # função de ativação
    
    zj = np.array(np.concatenate([np.ones(1), np.asarray(ha)]))
    
    y = W2.dot(zj)
    
    deltaK = y-t # erro * derivada de sigma(ak)
    
    Ev += deltaK.T.dot(deltaK)

msev.append(Ev/N)

msev = np.ravel(msev)
    
plt.plot(mse)