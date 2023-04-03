#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 19:28:39 2019

@author: vboas
"""

import matplotlib.pyplot as plt
import numpy as np

#n = 1000
#tx = 200
#w = 2 * np.pi / tx
#
#t1 = np.linspace(0, tx, n)

X = np.load('/mnt/dados/datasets/eeg_epochs/BCI4_2a/A01T.npy')

X = X[0,0,:,625:1125]

Fs =250 
T = 1/Fs # periodo de amostragem
n_samples = 2 * Fs
t = np.arange(0,n_samples) * T  # 
#t = np.linspace(0,2,n_samples) 
nf = Fs/2.0

# Função da onda senoidal = A*sen(2*pi*freq*t)

sinal1 = 20.0*np.sin(2.0*np.pi*2*t)
sinal2 = 20.0*np.sin(2.0*np.pi*4*t)
sinal3 = 20.0*np.sin(2.0*np.pi*9*t)
s = sinal1 + sinal2 + sinal3

bins = np.fft.fftfreq(n_samples, d=T) # d= período de amostragem (define a resoluo em frequencia - intervalo entre bins)
mascara = bins > 0 # apenas a parte positiva (retira espelhamento da fft)

fft_calculo = np.fft.fft(sum(X))
fft_abs = np.abs(fft_calculo)
fft_abs = 2.0*1000000*np.abs(fft_calculo/n_samples) # normalizar amplitude pelo num amostras e ajustar escala conforme dominio tempo

plt.figure(1)
plt.title("Sinal Original")
plt.plot(t,sum(X)*1000000)

plt.figure(3)
plt.title("Sinal da FFT")
plt.plot(bins[mascara],fft_abs[mascara])
plt.xticks(np.linspace(0,130,13, s))
