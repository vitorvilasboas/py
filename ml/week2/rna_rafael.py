#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:34:47 2019

@author: vboas
"""

import numpy as np

x = np.asarray([[1,0,0],[1,0,1],[1,1,0],[1,1,0]])

t = [0,1,1,0]

w1 = np.asarray([[0.3252,1.3703,-0.1022],[-0.7549,-1.7115,-0.2414]])

w2 = np.asarray([0.3192,0.3229,-0.8649])

# a = x.dot(w1.T) # produto vetorial (x*w1)

a = np.matmul(x,w1.T)

z = np.ones([4,3]) # acrescentando bias à camada oculta

z[:,1:3] = np.tanh(a) # camada oculta

#y = z.dot(w2.T)
y = np.matmul(z,w2.T)


yy = [3,4,5]

rr = [0,0,1]

sm = [np.exp(yy[i]) / np.sum(np.exp(rr)) for i in range(len(yy))]

c = - np.sum([np.log(sm[i]) for i in range(len(sm))])



v = np.zeros([4,1])
q = np.concatenate((v,z), axis=1)



