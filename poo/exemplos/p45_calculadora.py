"""
Crie um programa que simule uma calculadora comum, capaz de realizar as quatro operações aritméticas básicas,
a partir de dois números fornecidos pelo usuário. O programa deve ser capaz de realizar quantas operações o usuário quiser,
dando a opção ao mesmo de encerrar o programa.
"""


def soma(x, y):
    return x + y


def subtracao(x, y):
    return x - y


def multiplicacao(x, y):
    return x * y


def divisao(x, y):
    return x / y


def menu():
    opcao = input("""
  ############################
   Escolha uma operação: 
     [1] Soma
     [2] Subtração
     [3] Multiplicação
     [4] Divisão
     [outro] Sair
  ############################
  opção: 
  """)
    return opcao


def solicitaValores():
    a = int(input("Informe o 1o número inteiro: "))
    b = int(input("Informe o 2o número inteiro: "))
    return a, b


while True:
    operacao = menu()
    if operacao == '1':
        a, b = solicitaValores()
        resultado = soma(a, b)
    elif operacao == '2':
        a, b = solicitaValores()
        resultado = subtracao(a, b)
    elif operacao == '3':
        a, b = solicitaValores()
        resultado = multiplicacao(a, b)
    elif operacao == '4':
        a, b = solicitaValores()
        resultado = divisao(a, b)
    else:
        print("Saindo...")
        break;
    print(f"Resultado: {resultado}")