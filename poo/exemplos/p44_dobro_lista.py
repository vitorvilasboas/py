"""
Crie uma função que receba uma lista de números fornecidos pelo usuário e retorne uma nova lista cujos os elementos
são o dobro dos elementos na lista inicial.
"""


def calculoDobro(lista1):
    lista2 = list()
    for x in lista1:
        lista2.append(x ** 2)
    return lista2


# percorrendo lista
lista = list()
while True:
    num = int(input(
        "Informe um número inteiro. Para sair digite -99: "))  # flag (bandeira) para saida do laço de repetição é -99
    if (num != -99):
        lista.append(num)
    else:  # se o número informado for igual à -99
        print("Ok, você saiu do laço! ")
        break  # para a execução do laço no momento em que o flag foi informado

print(f"Lista inicial: {lista}")

quadrados = calculoDobro(lista)

print(f"Lista resultante: {quadrados}")