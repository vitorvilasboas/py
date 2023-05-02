# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:40:40 2019
@author: Vitor Vilas-Boas
"""
# Método range
range(5)

list(range(5))

# FOR - estrutura de repetição com variável e controle

# for iterador in range(inicio, fim, passo):
# limites são conhecidos

for i in range(5):
    print(i)

for i in range(0,10,1):
    pass

for i in range(10,0,-1):
    pass

for i in range(10,100,2):
    pass


# list comprehension
x = [1,2,3,4]

out = []
for item in x:
    out.append(item**2)
print(out)

[item**2 for item in x]


lista = ['A','B','C']  
for i in range(0, len(lista)):
    pass

# enumerate retorna indice e item
for pos, item in enumerate(lista): 
    print(f'A posição é {pos} e o item é {item}')


for a, b in zip(range(0, 5), range(5, 10)):
    print(a,'>>>',b)
    

X = [1,2,3,4]
out = [x**2 for x in X]
print(out)
    
    
# WHILE - estrutura de repetição com variável e controle   
    
# while (condicao):

# limites desconhecidos, mas flag conhecido  
c = 1
while(c<10):
    print(c, end=' ')
    c += 1
    
i = 1
while i < 5:
    print('i is: {}'.format(i))
    i = i+1


r = 'S'
n = list() 
while r == 's':
    n.append(int(input('Informe um número: ')))
    r = str(input('Continuar? '))
print('Fim')
print(n)



# limites desconhecidos, True break
while True:
   n = int(input('Informe um número: '))
   if n == 0:
       break

    
    
    
    
