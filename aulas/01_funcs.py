"""
Depurar: executar linha a linha para inspeção
Breakpoint define a linha onde a execução será interrompida para inspeção
Teclas de atalho:
    F8      executa linha a linha após primeiro breakpoint (Step over)
    Alt+F8  abre a view Evaluate expression (possibilita avaliar cada variável/objeto)
    F7      próximo passo dentro de um bloco de instrução ou função (Step into)
"""

a = 10
b = 1, 2, 3
x = [a, b]


def function1():
    print("qualquer texto")


def function2():
    function1()


def function3():
    function2()


def function4():
    f4 = 50
    function3()


def function5():
    function4()


function5()

print(a, b, x)

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