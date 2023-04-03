# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import numpy as np
import matplotlib.pyplot as plt

x_correct = np.linspace(0, 1, 100)
t_correct = np.sin(2 * np.pi * x_correct) # Função alvo, normalmente desconhecida, a qual deseja-se aproximar.

x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

plt.figure(figsize=(10,6))
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="dados de treinamento")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.ylabel('$t$', fontsize=16, rotation = 360)
plt.xlabel('$x$', fontsize=16)
plt.xticks([0,1])
plt.yticks([-1,0,1])
plt.legend()
#plt.savefig('figures/Fig1.1.1_.png', format='png', dpi=300, transparent=True, bbox_inches='tight')