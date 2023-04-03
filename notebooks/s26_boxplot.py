import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('datasets/trees.csv')

plt.boxplot(base.Volume)

#, vert = False, showfliers = False, notch = True,
#            patch_artist = True)
plt.title('Árvores')
plt.xlabel('Volume')

# colorir
# https://matplotlib.org/gallery/statistics/boxplot_demo.html

#plt.boxplot(base)
#
#plt.boxplot(base.Volume, vert = False)
#plt.boxplot(base.Girth, vert = False)
#plt.boxplot(base.Height, vert = False)