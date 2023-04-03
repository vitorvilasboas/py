# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import norm 

x = [8,2,10,10]
x = np.linspace(0,1,1000)
ruido = np.random.normal(scale=0.25, size=len(x)) # np.random.randn(len(x)) 
x = np.sin(2*np.pi*x) + 0.1*ruido

# Variância e desvio padrão são medidas de variabilidade e medem a dispersão (dos 
# dados em relação à média
# variancia = soma dos quadrados das diferenças entre cada observação e a média 
# das observações dividido pelo número de observações
var1 = np.sum(np.asarray(x-np.mean(x))**2)/len(x) 
var2 = np.var(x)

# desvio padrão = 
std1 = math.sqrt(var1)
std2 = np.std(x)

normal1 = ( 1/math.sqrt(2*np.pi*var1) ) * np.exp( (-1/(2*var1))*((x-np.mean(x))**2)  )

normal2 = norm.pdf(x, np.mean(x), np.std(x))