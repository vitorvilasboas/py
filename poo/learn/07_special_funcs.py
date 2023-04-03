# -*- coding: utf-8 -*-

#### Lambda — em tempo de execução — reduzidas
z = lambda b, e=2: b ** e

res = z(3, e=2)


#### Map — itera os valores passando por uma função específica
def func(b, e=2):
    return b ** e

lista = [1, 2, 3, 4, 5]

resultado = list(map(func, lista))
print(resultado)

resultado = list(map(lambda b, e=2: b ** e, lista))
print(resultado)


#### Filter — filtra valores em uma lista conforme uma condição
def par(x): return True if x % 2 == 0 else False

lista = [1, 2, 3, 4, 5, 6]

resultado = list(filter(par, lista))
print(resultado)

resultado = list(filter(lambda x: x % 2 == 0, lista))
print(resultado)


#### Reduce — Aplica uma função a uma lista 2 a 2

from functools import reduce

def media(x, y):
    return (x+y)/2

lista = [1, 2, 3, 4, 5, 10, 15, 20, 50]

resultado = reduce(produto, lista)  # produtório
print(resultado)

print(reduce(lambda x, y: x * y, [1, 2, 3, 4, 5]))


"""
Exemplo: Use expressões lambda e a função filter () para filtrar as palavras de uma lista que não começam com a letra 's'. Por exemplo: na lista ['sopa','cachorro','salada','gato','ótimo'] filtrar 'sopa' e 'salada'.
"""
lista = ['sopa', 'cachorro', 'salada', 'gato', 'ótimo']
print(list(filter(lambda palavra: palavra[0] == 's', lista)))
