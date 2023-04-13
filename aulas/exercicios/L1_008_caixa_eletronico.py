"""
Desenvolva um programa que simule a contagem de cédulas de um caixa eletrônico, perguntando ao usuário qual o valor a ser sacado (deve ser inteiro). Sabendo que o caixa possui cédulas de R$50, R$20, R$10 e R$1, o programa deve informar quantas cédulas de cada valor deverão ser entregues primando pelo menor número de cédulas possível. Utilize funções para cada tarefa necessária.
"""

def solicita():
    return int(input("Informe o valor que deseja sacar do caixa: R$"))

def conta_notas(x):
    n50 = x // 50
    x = x % 50
    n20 = x // 20
    x = x % 20
    n10 = x // 10
    x = x % 10
    return x, n10, n20, n50

if __name__ == '__main__':
    valor = solicita()
    n1, n10, n20, n50 = conta_notas(valor)
    print("Foi sacado um valor de R${:.2f} sendo {} notas de 50, {} notas de 20, {} notas de 10 e {} notas de 1 real.".format(valor, n50, n20, n10, n1))