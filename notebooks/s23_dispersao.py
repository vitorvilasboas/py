import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base = pd.read_csv('trees.csv')

plt.scatter(base.Girth, base.Volume, color = 'blue', facecolors = 'none', marker = '*')
plt.title('Árvores')
plt.xlabel('Volume')
plt.ylabel('Circunferência')

plt.plot(base.Girth, base.Volume)

sns.regplot(base.Girth, base.Volume, data = base, x_jitter = 0.3, fit_reg = False)