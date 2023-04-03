# -*- coding: utf-8 -*-
# ========= Linear Models for Classification ==================================
# Considere o dadaset IRIS, com três classes, C1, C2 e C3. Fa¸ca:
# 1. Com base na Seção 4.1.4, determine o vetor de w, dado pela Eq. 4.30, para os seguintes cenários:
#       1.1 A classe C1 versus a classe C2
#       1.2 A classe C1 versus a classe C3
#       1.3 A classe C2 versus a classe C3
# 2. Com base na Seção 4.1.6, determine a matriz W, dado pela Eq. 4.51, para o caso multiclasse.
# Esboce os resultados dos item 1 e 2 através de gráficos como a Figura 4.5. Comente os resultados
# =============================================================================

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


