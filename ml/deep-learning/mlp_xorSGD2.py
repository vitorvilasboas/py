# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:24:30 2019

@author: Rafael Rocha
"""

#%% Pacotes utilizados
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import keras.backend as kb
from sklearn.metrics import mean_squared_error

#%% Funções
def sse (t, y):
    
    """
    Sum-of-squares error
    """
    
    error = kb.square(y - t)
    
    return error

def mse (t, y):
    
    """
    Mean squared error
    """
    
#    error = kb.mean((kb.sqrt(y-kb.transpose(t))))
    error = kb.mean(kb.square(y - t), axis=-1)
    
    return error


#%% Parâmetros iniciais
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = np.array([[0],[1],[1],[0]])

D = x.shape[1]
M = 5
K = 1

print(np.concatenate([x,t], axis=1))

#%% Criação da rede
model = Sequential()

model.add(Dense(M, input_shape=(D,), activation='tanh'))
model.add(Dense(K, activation='sigmoid'))

initial_weights = model.get_weights()

model.compile(SGD(lr=0.5),
#              loss='mean_squared_error'
              loss=[sse]
              )
h = model.fit(x, t, epochs=500, verbose=0)

model.summary()

#%% Plots
y = model.predict(x)

x_temp = np.arange(1, np.size(h.history['loss'])+1)

plt.figure()
plt.plot(x_temp, h.history['loss'])
#plt.plot(x_temp, h.history['val_loss'])
plt.ylabel('EQM')
plt.xlabel('Época')
#plt.legend(['Treinamento', 'Validation'])

#plt.figure()
#plt.plot(t,'o', y, 'o')
#plt.xlabel('Exemplos')
#plt.ylabel('Saídas')
#plt.legend(['Saída esperada', 'Saída da rede'])
##plt.grid()
#plt.axhline(0.5, color='black')

print('\n')
print(mean_squared_error(t, y))