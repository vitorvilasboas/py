"""
Exemplo: Crie um programa Python capaz de simular o funcionamento de uma calculadora ao ler valores inteiros ou reais e a operação desejada pelo usuário. As operações, implementadas via funções , devem ser: soma, subtração, multiplicação, divisão, potenciação, raiz, divisão inteira, resto da divisão e logaritmo.
- Eventuais restrições nas operações devem ser consideradas (ex.: divisão por zero)
- O programa deve perguntar ao usuário se deseja realizar outra operação ou sair da execução (crie um menu).
"""

def soma(a, b):
    return a + b

def subtracao(a, b):
    return a - b

def produto(a, b):
    return a * b

def divisao(a, b):
    return a / b

# def divisao(a, b):
#   if b != 0:
#     return a / b
#   else:
#     return "Não é possível divisão por zero!!"

def resto(a, b):
    return a % b

def divi_inteira(a, b):
    return a // b

def potencia(base, expoente):
    import math
    pot = math.pow(base, expoente)
    # pot = base ** expoente
    return pot

def raiz(radicando, indice):
    import math as m
    raiz = m.pow(radicando, 1 / indice)
    # raiz = radicando ** (1/indice)
    return raiz

def logaritmo(logaritmando, base):
    from math import log
    return log(logaritmando, base)

def solicita():
    num1 = float(input("\nInforme o 1º número: "))
    num2 = float(input("Informe o 2º número: "))
    return num1, num2

def menu():
    print("\n====================================")
    opcao = input("""Escolha uma operação: 
  [1] Soma
  [2] Subtração
  [3] Multiplicação
  [4] Divisão
  [5] Potenciação
  [6] Raiz
  [7] Logaritmo
  [0] Sair

  operação: """)
    return opcao

def delay():
    import time
    time.sleep(1)
    print(".", end="")
    time.sleep(1)
    print(".", end="")
    time.sleep(1)
    print(".")
    time.sleep(1)

import time

print("#####################")
print("###### MY CALC ######")
print("#####################")

while (True):
    operacao = menu()
    if operacao == '0':
        confirme = input("\nRealmente deseja sair? [S]sim [N]não : ")
        if confirme.upper() == 'S':
            break
    else:
        time.sleep(2)
        x, y = solicita()
        if operacao == '1':
            r = soma(x, y)
        elif operacao == '2':
            r = subtracao(x, y)
        elif operacao == '3':
            r = produto(x, y)
        elif operacao == '4':
            if y != 0:
                print(f"Divisão {x}/{y} = {divisao(x, y)}")
                print(f"Divisão inteira {x}//{y} = {divi_inteira(x, y)} (resto = {resto(x, y)})")
            else:
                print("\nOps!! Indeterminação encontrada (divisão por zero)! Tente novamente!")
            continue
        elif operacao == '5':
            r = potencia(x, y)
        elif operacao == '6':
            # condição: radicando positivo e índice não nulo
            if x >= 0 and y != 0:
                r = raiz(x, y)
            else:
                print("\nOps!! Violação da condição de existência da raiz! Tente novamente!")
                continue
        elif operacao == '7':
            # logaritmo existe se: logaritmando positivo E base positiva E diferente de 1.
            if (x > 0) and (y > 0 and y != 1):
                r = logaritmo(x, y)
            else:
                print("""\nOps!! Condição de existência do logaritmo violada. 
              Logaritmando deve ser positivo e a base deve ser positiva e diferente de 1! 
              Tente novamente!""")
                continue
        else:
            print("Ops!! Operação inválida, tente novamente", end="")
            delay()
        print(f"\nResultado = {round(r, 1)} \n")
print("\nEncerrando o programa", end="")
delay()
print("\n###### Fatality!! ######")