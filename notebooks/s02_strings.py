# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:36:51 2019
@author: Vitor Vilas-Boas
"""
#-%% Métodos strings

st = 'Olá, prazer, me chamo loriosmenilson!'

len(st) # comprimento (num caracteres)

st.lower() # transforma - todos minusculos
st.upper() # ...todos maiusculos
st.capitalize() # ...somente primeira maiuscula
st.title() # ...somente inicios de palavras em maiusculo
st.count('o') # conta 
st.find('prazer') # retorna a posição em que o termo inicia (-1 se nao foi encontrado)

teste = 'prazer' in st # termo está na string? True ou False
teste = 'prazer' not in st


st.replace('Olá','Oi') # substitui termos

st.split(',') # divide string nas virgulas
st.split() # divide a string nos espaços

'-'.join(st) # junta novamente a string com - no lugar dos espaços

'  Aprenda  '.strip() # elimina espaços indesejados
'  Aprenda  '.rstrip() # elimina espaços indesejados à direita
'  Aprenda  '.lstrip() # elimina espaços indesejados à esquerda


Num = '32'
Rua = 'Joaquim N6'

# Métodos is (True ou False)
print(st.isalpha()) # é alfabético?
print(Rua.isnumeric())  # é numérico?
print(Rua.isalnum()) #  alfanumrico?

mundial = 'Dale Verdão! #Palmeiras'
termos = mundial.split('#')
print(termos[1])





