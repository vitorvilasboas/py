#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

M = 9

A_train = np.asarray([x_train**i for i in range(M+1)]).T 

l = -18

w = np.linalg.solve(A_train.T @ A_train, A_train.T @ t_train) # option one
w2 = np.linalg.inv(A_train.T @ A_train) @ (A_train.T @ t_train) # w1 = np.linalg.inv(A_train.T.dot(A_train)).dot(A_train.T.dot(t_train))

# w3 = np.linalg.pinv(A_train) @ t_train # inv(A.T * A) * A.T * t  # without regularization


wl = np.linalg.solve( np.exp(l) * np.eye(np.size(A_train, 1)) + A_train.T @ A_train, A_train.T @ t_train) # option one

wl1 = np.linalg.inv(A_train.T.dot(A_train) + np.exp(l) * np.identity(M+1)).dot(A_train.T.dot(t_train)) # option two