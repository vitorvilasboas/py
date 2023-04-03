# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np

data = {'Nomes':['Fulano','Ciclano','Beltrano'], 'Idade':[10,32,58]}

dataf = pd.DataFrame(data)

data = list([['Fulano',10], ['Ciclano',32], ['Beltrano',58]])

dataf = pd.DataFrame(data, columns=['Nome', 'Idade'])

dados = pd.read_csv('./../5. Atividade/credito.csv')
#dados = pd.read_csv(os.path.dirname(__file__) + '/../5. Atividade/credito.csv')

dados.shape

dados.head

dados.describe


#Fatiamentos Dataframe - slice

#loc obtém linhas (ou colunas) com rótulos específicos do índice.
more30 = dados.loc[dados['age'] > 30]

#iloc obtém linhas (ou colunas) em posições específicas no índice (portanto, são necessários números inteiros).
# com testes booleanos
more30 = dados.loc[(dados['age'] > 30).values]
more30 = dados.iloc[np.where(dados['age'] > 30)]
more30 = dados.iloc[(dados['age'] > 30).nonzero()]
more30 = dados.iloc[(dados['age'] > 30).nonzero()]

idx = np.asarray((dados.iloc[:,0] > 30).nonzero())

dataf = dataf.rename(columns={'Nome':'Name', 'Idade':'Age'})

dataf['Age'].value_counts() # conta o número de ocorrência de cada registro único

dataf.drop([0]) # excluir linha

dataf.drop('Name', axis=1) # excluir coluna
dataf.drop(columns=['Name']) # excluir coluna

x = np.linspace(0,100,1000)

x1 = x.reshape(250,4)

objeto = pd.Series(x1) #Dataframe de uma coluna só