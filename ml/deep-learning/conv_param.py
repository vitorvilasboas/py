# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:10:52 2019

@author: Rafael Rocha
"""

# Parâmetros camada convolucional
n_channels = 1
featureMaps = 6
kernels = (5,5)

paramsConv = featureMaps*((n_channels*(kernels[0]*kernels[1]))+1)

print('Camada convolucional: ' + str(paramsConv))

# Parâmetros MLP
neurons = 120
featureMaps = 16
featureMapSize = (5,5)

vec = 16*(featureMapSize[0]*featureMapSize[1])
print('Vetor de características: ' + str(vec))

paramsMLP = neurons*(vec+1)
print('MLP: ' + str(paramsMLP))