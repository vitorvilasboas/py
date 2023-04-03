# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:22:45 2019

@author: Rafael Rocha
"""

import keras
from keras import backend as K
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

input_shape = (32, 32, 1)

K.clear_session()

model = keras.Sequential()

# Lenet5
model.add(Conv2D(6, kernel_size=5, strides=1,
                 padding='valid', activation='tanh',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(16, kernel_size=5, strides=1, padding='valid',
                 activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))
# Lenet5

print('Lenet5:')
model.summary()