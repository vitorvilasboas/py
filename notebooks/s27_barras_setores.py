import pandas as pd

base = pd.read_csv('datasets/insect.csv')

agrupado = base.groupby(['spray'])['count'].sum()

#agrupado.plot.bar(color = 'gray')

agrupado.plot.pie(legend = True)