# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:40:40 2019
@author: Vitor Vilas-Boas
"""

def func1(n1,n2):
    return n1+n2

teste = func1(2,3)
print(teste)


def my_func(param1='default'):
    """
    Documentação da função vai aqui. #docstring
    """
    print(param1)

my_func

my_func()

my_func('novo parametro')

my_func(param1='novo parametro')

help(my_func)
print(my_func.__doc__)

def square(x):
    return x**2
out = square(x=2)

print(out)


def vezes2(var):
    return var**2

vezes2(2)

# funções Lamda - em tempo de execução - reduzidas 
z = lambda var,e: var**e

z(3,2)


# escopo de variáveis
def func(b):
    # global n # utiliza a variável global
    n = b**2
    return n

n = 2

x = func(n)

print(n,x)

try:
    xx = func2(n)
except:
    print('Falha')
else:
    print(xx)
finally:
    print('Exibe sempre')
    
    
# Função Map - itera os valores passando por uma funço especfica
seq = [1, 2, 3, 4, 5]

list(map(func, seq))

list(map(lambda x:x**2, seq))



# Filter
def par(x): return (True if x % 2 == 0 else False)

list(filter(par, seq))

list(filter(lambda x: x % 2 == 0, seq))



# Reduce - Aplica uma função a uma lista 2 a 2
from functools import reduce
def soma(x, y): return x + y
def produto(x, y): return x * y
    
reduce(soma, seq) # somatório
reduce(produto, seq) # produtório

reduce(lambda x,y : x + y, seq)
reduce(lambda x,y : x * y, seq)

"""
Existem algumas situações aonde queremos que nosso código seja executado 
apenas sob condições especiais, é o caso dos módulos principais. 
Só queremos que nossa função main() seja executada se o módulo for o principal. 
Caso ele tenha sido importado, a aplicação só deverá ser executada se main() 
for chamado explicitamente.
"""

def main():
    """Função principal da aplicação"""
    print('Vitor')

if __name__ == "__main__":
    main()