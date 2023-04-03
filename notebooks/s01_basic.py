# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:36:51 2019
@author: Vitor Vilas-Boas
"""
# comentário em linha

'''
comentário em bloco
'''

print("Bem vindo!")

print("Todos os comando são vistos como funções no python 3.x")

# Tipos primitivos (int, float, bool, str)
a = 1   # int
b = 2   # int
c = 7.  # float
var = 'texto'   # string - Aspas simples é recomendado (boa prática)
Var = "Case sensitive"  # string
teste = True    # booleano

type(c)

# operações básicas - aritméticas
resultado = a + a - b * c
exp = 3**2
exp = pow(3,2)
div = 3 / 2
divint = 3 // 2
preced = (3 - 2) * 5
resto = 6 % 5

conctStr = '8' + '3'

print('Oi'*20) 

# Conversão entre tipos
resultado = int(div)
exp = float(exp)
preced = str(preced)
teste2 = bool(1) # 1 = True
teste2 = bool(0) # 0 = False
teste2 = bool() # quando nenhum argumento é informado, asuume False
teste2 = bool(2) # para qualquer valor diferente de 0, assume True

#-%% Blocos
print("Isto é um bloco?")
print("\nÉ SIM  ")

#-%% Entrada/Saida
print(input("Qual seu nome? "))

name = input('Nome: ') 
surname = input('Sobrenome: ') 

print('Nome: ', name)
print('Nome é ' + name + ' e sobrenome é ', surname)

print('Resultado: ' + resultado) # error - o uso é permitido somente para concatenação de strings
print('Resultado: ' + str(resultado))  # solução converter para string
print('Resultado: ', resultado) # ou usar virgula


# Saída com Format
print("Meu nome é {}".format(name))

idade = 99
print("Meu nome é {n}, tenho {m} anos!".format(n=name,m=idade))

print("Meu nome é {0}, tenho {1} anos!".format(name,idade))

print("Meu nome é {:^20}".format(name)) # centralizado em 20 espaços
print("Meu nome é {:<20}".format(name)) # à esquerda em 20 espaços
print("Meu nome é {:<20}".format(name)) # à direita em 20 espaços

print(f'Minha idade é {idade}') # fstrings a partir do py 3.6 # ativa inrpolação em strings

# Resumo
n1 = int(input('Informe um valor: '))
n2 = int(input('Informe outro valor: '))
s = n1 + n2
m = n1 * n2
d = n1 / n2
di = n1 // n2
e = n1 ** n2
print('A soma é {}, \n o produto é {} e a divisão é {:.2f}'.format(s,m,d), end='')
print('A divisão inteira é {} e a potência é {}'.format(di,e))

print('A soma é %d, \n o produto é %d e a divisão é %.2f' % (s, m, d))