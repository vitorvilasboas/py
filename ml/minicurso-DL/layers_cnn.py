# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:54:13 2019

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
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

data = np.load('dataset_mnist.npz')
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
y_train = to_categorical(y_train)

D = (np.size(x_train, 1), np.size(x_train, 2),  np.size(x_train, 3))

#%% Criação da rede
K.clear_session()
model = Sequential()

model.add(Conv2D(5, kernel_size=3, strides=1, padding='same', input_shape=D))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(10, kernel_size=3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=SGD(lr=0.001),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

#%% Mapa de características

output_tensor = K.function([model.layers[0].input], [model.layers[0].output])
layer_output = output_tensor([x_train])
out = np.array(layer_output)
x_train_conv1 = out[0,:]

output_tensor = K.function([model.layers[0].input], [model.layers[1].output])
layer_output = output_tensor([x_train])
out = np.array(layer_output)
x_train_act1 = out[0,:]

output_tensor = K.function([model.layers[0].input], [model.layers[2].output])
layer_output = output_tensor([x_train])
out = np.array(layer_output)
x_train_pool1 = out[0,:]

output_tensor = K.function([model.layers[0].input], [model.layers[3].output])
layer_output = output_tensor([x_train])
out = np.array(layer_output)
x_train_conv2 = out[0,:]

output_tensor = K.function([model.layers[0].input], [model.layers[4].output])
layer_output = output_tensor([x_train])
out = np.array(layer_output)
x_train_act2 = out[0,:]

output_tensor = K.function([model.layers[0].input], [model.layers[5].output])
layer_output = output_tensor([x_train])
out = np.array(layer_output)
x_train_pool2 = out[0,:]

#%% Visualizar mapas de características
n = 1

plt.figure()
plt.subplot(141)
imshow(x_train[n,:,:,0], cmap='gray')

plt.subplot(142)
imshow(x_train_conv1[n,:,:,0], cmap='gray')

plt.subplot(143)
imshow(x_train_act1[n,:,:,0], cmap='gray')

plt.subplot(144)
imshow(x_train_pool1[n,:,:,0], cmap='gray')

plt.figure()
plt.subplot(141)
imshow(x_train[n,:,:,0], cmap='gray')

plt.subplot(142)
imshow(x_train_conv2[n,:,:,4], cmap='gray')

plt.subplot(143)
imshow(x_train_act2[n,:,:,4], cmap='gray')

plt.subplot(144)
imshow(x_train_pool2[n,:,:,4], cmap='gray')