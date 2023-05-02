#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:55:27 2019
@author: vboas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plt.figure()
colors = ['navy', 'turquoise', 'darkorange']

plt.rcParams.update({'font.size': 12})

# %% gerar curva contnua através dos pontos de cordenada (x,y)
plt.figure(figsize=(8,5))
x = np.linspace(0,1,100)
y = np.sin(2*np.pi*x) # f(x)
g = np.cos(2*np.pi*x)
plt.plot(x,y, color=colors[2], ls='-.', linewidth=5, label='sen(x)')
plt.plot(x,g, color=colors[1], ls='-.', label='cos(x)')
plt.legend(loc='upper right')
plt.xlim(-0.02, 1.02)
plt.ylim(-1.1, 1.1)
plt.xticks([0, 0.5, 1])
plt.xticks([0, np.pi/2, np.pi, 3/2*np.pi, 2*np.pi], 
           ['0', r'$\frac{1}{2}\pi$',r'$pi$',r'$\frac{3}{2}\pi$',r'$2*pi$']) # integrando com Latex ([posições],[valores])
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gráfico')
plt.grid(True)
plt.savefig('figura.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight')


# %% gerar pontos no gráfico de cordenadas (x,t) - gráfico de dispersão
x = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])
f = np.sin(2*np.pi*x)
plt.scatter(x,t, color=colors[0], alpha=.8, lw=2, label='teste', edgecolors='face', 
            facecolor=None, **dict(marker='o'), cmap=None, s=20, norm=None, vmin=None, 
            vmax=None, verts=None) # cmap mapa de cores; alpha=opacidade; faceted=True, , hold=None
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('titulo')


# %%
x=[0,1,2,3,4]
y=[2,3,5,1,0]

plt.figure(figsize=(8,5))

plt.plot(x,y, LineWidth=4, marker='v', 
                             markersize=15, 
                             markeredgecolor='k', 
                             markeredgewidth=3,
                             markerfacecolor='g')
plt.xlabel('x')
plt.ylabel('y')


# %%
x = np.random.rand(20)
y = np.random.rand(20)
plt.figure(figsize=(10,7))
plt.scatter(x,y, color='b', s=200, alpha=0.60)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xticks(np.linspace(0,1,5))
plt.yticks(np.linspace(0,1,5))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gráfico aleatório de dispersão')
plt.grid(True)


# %%
x = np.arange(3)
y = np.random.rand(3)
plt.bar(x,y, color=colors, tick_label=['Barra A', 'Barra B', 'Barra C'])
plt.ylim(0,1)
plt.yticks(np.linspace(0,1,5))


# %% carregando arquivos.csv

data = pd.read_csv('./pop.csv')

data.head() # imprime o sumário do dataframe

data[["Ano","Brasil","Sudeste"]].plot()

ax = data[["Brasil","Sudeste"]].plot(figsize=(8,5), kind='bar')
ax.set_xticklabels(data["Ano"])



