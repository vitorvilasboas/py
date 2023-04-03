# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# F(x): Função alvo - Para fins didaticos é conhecida, mas normalmente é desconhecida! É o que deseja-se aprender.
x_correct = np.linspace(0, 1, 1000)
t_correct = np.sin(2 * np.pi * x_correct)

## Dataset Treinamento com rótulos=alvo=target -> mapeamento de x para y (x --> y)
x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])

plt.figure(figsize=(10,6))
plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
plt.ylabel('$t$', fontsize=16, rotation = 360)
plt.xlabel('$x$', fontsize=16)
plt.xticks([0,1])
plt.yticks([-1,0,1])
plt.legend()
plt.savefig('figures/Fig1.1.png', format='png', dpi=300, transparent=True, bbox_inches='tight')