# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:14:35 2019

@author: Rafael Rocha
"""

#%% Pacotes utilizados
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.datasets import mnist, cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import  load_digits
from skimage.io import imread, imshow
from skimage import img_as_ubyte, img_as_float

#%% Parâmetros iniciais
#digits = load_digits()
#
#img = digits.images[8]
#x = digits.images
#t = digits.target
#target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

data = np.load('dataset_digits.npz')
x = data['x_images']
t = data['t']
img = data['img']
target_names = data['target_names']

x_train, x_test, t_train, t_test = train_test_split(x, t,
                                                        test_size=0.3,
                                                        stratify=t)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

t_train_cat = to_categorical(t_train)
t_test_cat = to_categorical(t_test)

D = (np.size(x_train, 1), np.size(x_train, 2),  np.size(x_train, 3))

#%% Criação da rede
K.clear_session()
model = Sequential()

model.add(Conv2D(5, kernel_size=3, strides=1, padding='same', activation='relu',
                 input_shape=D))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(10, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=SGD(lr=0.001),
              metrics=['accuracy'])

h = model.fit(x_train, t_train_cat, epochs=500, verbose=1,
              validation_data = (x_test, t_test_cat))

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