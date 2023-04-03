# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:18:11 2019

@author: Rafael Rocha
"""

#%% Pacotes utilizados
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#%% Parâmetros iniciais
data = np.load('dataset_banknotes.npz')
x = data['x']
t = data['t']

x_train, x_test, t_train, t_test = train_test_split(x, t,
                                                        test_size=0.2,
                                                        stratify=t)

D = x.shape[1]
M = 5
K = 1

#%% Criação da rede
model = Sequential()

model.add(Dense(M, input_shape=(D,), activation='tanh'))
#model.add(Dense(M, activation='tanh'))
model.add(Dense(K, activation='sigmoid'))

initial_weights = model.get_weights()

model.compile(SGD(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
h = model.fit(x_train, t_train, epochs=100, verbose=1,
              validation_data = (x_test, t_test))

#model.summary()

#%% Resultados
y = model.predict(x_test)
y_class = model.predict_classes(x_test)


print('\n')

target_names = ['Legítima', 'Falsificada']

cr = classification_report(t_test,
                           y_class,
                           target_names=target_names,
                           digits=4)
print(cr)

print('\nConfusion matrix:\n')
cm = confusion_matrix(t_test,
                      y_class)

print(cm)

#%% Plots
x_temp = np.arange(1, np.size(h.history['acc'])+1)

plt.figure()
plt.plot(x_temp, h.history['acc'])
plt.plot(x_temp, h.history['val_acc'])
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Teste'])

plt.figure()
plt.plot(x_temp, h.history['loss'])
plt.plot(x_temp, h.history['val_loss'])
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Teste'])