import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = pd.read_csv('datasets/trees.csv')

x = base.iloc[:,1]

h = np.histogram(x, bins = 2)

plt.hist(x, bins = 6)
plt.title('Árvores')
plt.ylabel('Frequência')
plt.xlabel('Altura')