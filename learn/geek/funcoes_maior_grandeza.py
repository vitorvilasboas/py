"""
Funções de Maior Grandeza (Higher Order Functions - HOF)
funções que retornam outras funções
passagem de funções como argumentos
variáveis do tipo de funções

Em Python, as funções são Cidadãos de Primeira Classe (First Class Citizen)

def somar(a, b):
    return a + b


def diminuir(a, b):
    return a - b


def multiplicar(a, b):
    return a * b


def dividir(a, b):
    return a / b


def calcular(num1, num2, funcao):
    return funcao(num1, num2)

print(calcular(4, 3, somar))
print(calcular(4, 3, diminuir))
print(calcular(4, 3, multiplicar))
print(calcular(4, 3, dividir))

# Nested Functions (Funções Aninhadas) ou Inner Functions (Funções Internas)
# funções dentro de funções
from random import choice
def cumprimento(pessoa):
    def humor():
        return choice(('E ai ', 'Suma daqui ', 'Gosto muito de você '))
    return humor() + pessoa

print(cumprimento('Angelina'))
print(cumprimento('Felicity'))

# Retornando funções de outras funções
from random import choice
def faz_me_rir():
    def rir():
        return choice(('hahahahahaha', 'kkkkkkkkkk', 'yayayayayayayaya'))
    return rir

rindo = faz_me_rir()
print(rindo())
"""

# Inner functions podem acessar o escopo de funções mais externas
from random import choice
def fazer_rir(pessoa):
    def rir():
        risada = choice(('hahahahaha', 'lolololololo', 'kkkkkkkkkkk'))
        return f'{risada} {pessoa}'
    return rir

rindo = fazer_rir('Vitor')
print(rindo())
print(rindo())
print(rindo())


