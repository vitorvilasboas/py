# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:20:25 2019
@author: Vitor Vilas-Boas
"""

# https://docs.scipy.org/doc/numpy/index.html
# NumPy (ou Numpy) é uma biblioteca de álgebra linear para Python

import numpy as np

lista = []

minha_lista = [1,2,3]
minha_lista


minha_lista = [1,2,3]
minha_lista

np.array(minha_lista)

minha_matriz = [[1,2,3],[4,5,6],[7,8,9]]
minha_matriz

np.array(minha_matriz)

# ## Métodos incorporados (Built-in Methods)

# ### arange
# Retorna valores uniformemente espaçados dentro de um determinado intervalo.

np.arange(0,10)

np.arange(0,11,2)

# ### zeros e ones
# Gerar matrizes de zeros ou de ums

np.zeros(3)

np.zeros((5,5))

np.ones(3)

np.ones((3,3))

# ### linspace
# Retorna números uniformemente espaçados ao longo de um intervalo especificado.

np.linspace(0,10,3)

np.linspace(0,10,50)

# ## eye 
# Cria uma matriz identidade
np.eye(4)

# ## Random 
# Numpy também tem muitas maneiras de criar arrays de números aleatórios:

# ### rand
# Cria uma matriz da forma dada e preencha com amostras aleatórias de uma distribuição uniforme sobre ``[0, 1)``.

np.random.rand(2)

np.random.rand(5,5)

# ### randn
# Retorna uma amostra (ou amostras) da distribuição "normal". Ao contrário de rand, que é uniforme:

np.random.randn(2)

np.random.randn(5,5)


# ### randint
# Retorna inteiros aleatórios de "low" (inclusive) para "high" (exclusivo).

np.random.randint(1,100)

np.random.randint(1,100,10)


# ## Atributos de Array e Métodos
arr = np.arange(25)
ranarr = np.random.randint(0,50,10)

# ## Reshape
# Retorna uma matriz contendo os mesmos dados com uma nova forma.

arr.reshape(5,5)
# ### max,min,argmax,argmin
# são métodos úteis para encontrar valores máximos ou mínimos, ou para encontrar seus locais de índice usando argmin ou argmax

ranarr

ranarr.max()

ranarr.argmax()

ranarr.min()

ranarr.argmin()


# ## Shape
# Shape é um atributo que os arrays têm (não um método):

# Vector
arr.shape

arr.reshape(1,25)

arr.reshape(1,25).shape

arr.reshape(25,1)

arr.reshape(25,1).shape


# ### dtype
# pegar o tipo de dados do objeto na matriz:
arr.dtype


a = np.array([[1,2,3],[2,5,8]])
b = np.array([[3,1,2],[2,2,4]])

np.r_[a,b] # é equivalente à np.concatenate([a,b])

np.hstack(a,b)