"""
Leia dois números inteiros e sorteie um número qualquer 
no intervalo entre os números fornecidos.
"""

import random

num1 = int(input("Digite o primeiro número inteiro: "))
num2 = int(input("Digite o segundo número inteiro (maior que o primeiro): "))

sorteado = random.randint(num1, num2)

print("Numero sorteado: {}".format(sorteado))
# print(f"Numero sorteado: {sorteado}")
# print("Numero sorteado: ", sorteado)