# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:40:40 2019
@author: Vitor Vilas-Boas
"""
# Variáveis compostas 

#-%% Tuplas - são imutáveis
t = (1,2,3) # parenteses são opcionais
t[0]
t[0] = 'NEW' # error

a = (2,4,9)
b = (5,9,6,2)
c = b + a
print(c)
print(c.count(2))
print(c.index(2))

print(sorted(c))

pessoa = ('Gustavo', 78, 'M', 88.77)

del(pessoa) # excluir tupla


#-%% Listas - permitem o armazenamento de objetos de diferentes tipos - MUTVEIS

list(c)

lista = [1,2,3]

lista2 = ['a','b','c']

type(lista)

lista = [1.0,7,'nome',[1, 2, 3]]

# A indexação do python começa em 0

print(lista[0])

print(lista[-1]) # último
print(lista[-2]) # penúltimo

lista[0]
lista[1]
lista[1:] # fatiamento a partir de __
lista[:1] # fatiamento até __
lista[1:3] # fatiamento a partir de __ até __
lista[2] = 'Vitor'

lista.append('d') # Add elementos no fim dalista

lista.insert(0,'opa') # add elemento em uma posição especifica

print(lista)

mylist = [1, 2, 3, [4 ,5 ,['teste']]]
mylist[3]
mylist[3][2]
mylist[3][2][0]

del mylist[2] # elimina elemento pela chave(indice)
mylist.pop(3) # elimina elemento pela chave(indice)
mylist.pop() #elimina o ltimo elemento
mylist.remove('2') #elimina elemento pelo valor

mylist.clear() #limpa lista

# criar listas com range
numbers = list(range(3,8))

numbers = [9,2,5,1,0,3,4]

print(numbers)
numbers.sort() # ordena crescente
print(numbers)
numbers.sort(reverse=True) # ordena decrescente 
print(numbers)
numbers.reverse() # ordena decrescente
print(numbers)

min(numbers)
max(numbers)
sum(numbers)

A = [5, 3, 4, 1]
# B = A # cria uma ligação entre A e B - interdependência
B = A[:] # cria uma cópia de uma lista em outra lista - independentes
print(f'Lista A: {A}')
print(f'Lista B: {B}')

B[2] = 7
print(f'Lista A: {A}')
print(f'Lista B: {B}')


## lista dentro de listas






# strings são listas de caracteres, logo pode-se acessar seus indices
string = 'minicurso de python com anaconda'
print(string[4:])
print(string[4:19:2]) # fatiamento a partir de __ até __ salto __
print(string[4::3]) # fatiamento a partir de __ salto __


#-%% Dicionários - indices literais

d = dict{}
d = {}
d = {'chave1':'item1','chave2':'item2'}
d['chave1']

d = {'name':'Vitor','surname':'Vilas-Boas'}
d['sexo':'M']

del d['sexo']

filme = {'titulo':'Star Wars',
         'ano':'1977',
         'diretor':'George Lucas'
        }

print(d.keys())
print(d.values())
print(d.items())


estado = dict()
brasil = list()

estado['uf'] = 'PA'
estado['Regiao'] = 'Norte'
brasil.append(estado)
#brasil.append(estado.copy())

estado['uf'] = 'SP'
estado['Regiao'] = 'Sudeste'
brasil.append(estado)
#brasil.append(estado.copy())

print(brasil)



#-%% Sets - coleção não ordenada e não indexada
{1,2,3}
{1,2,3,1,2,1,2,3,3,3,3,2,2,2,1,1,2}

thisset = {"apple", "banana", "cherry"}
print(thisset)

thisset.add("orange")

thisset.remove("banana")

thisset.discard("banana")

x = thisset.pop()

thisset.clear()

del thisset

thisset.update(["orange", "mango", "grapes"])

set1 = {"a", "b" , "c"}
set2 = {1, 2, 3}

set3 = set1.union(set2)
print(set3)




