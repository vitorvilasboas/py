"""Exemplo: Calcule o fatorial de um número fornecido pelo usuário"""

from math import factorial
x = input('Informe um número inteiro: ')
x = int(x)
fat = factorial(x) # 5! = 5 * 4 * 3 * 2 * 1 = 120
print(f'O fatorial de {x} é {fat}.')