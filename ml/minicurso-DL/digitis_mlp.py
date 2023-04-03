# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:52:25 2019

@author: Rafael Rocha
"""

#%% Pacotes utilizados
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as B
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import  load_digits

#%% Parâmetros iniciais
#digits = load_digits()
#
#img = digits.images[8]
#x = digits.data
#t = digits.target
#target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

data = np.load('dataset_digits.npz')
x = data['x_data']
t = data['t']
img = data['img']
target_names = data['target_names']

x_train, x_test, t_train, t_test = train_test_split(x, t,
                                                        test_size=0.3,
                                                        stratify=t)

t_train_cat = to_categorical(t_train)
t_test_cat = to_categorical(t_test)

D = x.shape[1]
M = 10
K = 10

#%% Criação da rede
B.clear_session()
model = Sequential()

model.add(Dense(M, input_shape=(D,), activation='tanh'))
model.add(Dense(M, activation='tanh'))
#model.add(Dense(M, activation='tanh'))
model.add(Dense(K, activation='softmax'))

initial_weights = model.get_weights()

model.compile(SGD(lr=0.05), 'categorical_crossentropy', metrics=['accuracy'])
h = model.fit(x_train, t_train_cat, epochs=500, verbose=1,
              validation_data = (x_test, t_test_cat))

#model.summary()

#%% Resultados
y = model.predict(x_test)
y_class = model.predict_classes(x_test)

print('\n')

train_acc = model.evaluate(x_train, t_train_cat, verbose=0)[1]
test_acc = model.evaluate(x_test, t_test_cat, verbose=0)[1]

print('Acurácia de treino:', train_acc)
print('Acurácia de teste:', test_acc)

print('\n')

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
plt.imshow(img, cmap='gray')

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
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Teste'])