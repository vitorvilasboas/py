"""
Crie um programa que faça o computador "pensar" em um número
inteiro entre 1 e 10 e peça para o usuário tentar adivinhar
qual foi o número escolhido pelo computador até acertar, ao
final mostre quantos palpites foram necessários até adivinhar.
"""

import random  # importando a biblioteca random
import time

comput = random.randint(1, 10)  # escolhe aleatoriamento um numero inteiro entre 1 e 10
tentativas = 0
while True:  # != é o sinal de diferença
    usuario = int(input("Pensei em um número inteiro entre 1 e 10. Tente adivinhar: "))
    tentativas += 1
    if usuario == comput:
        print("Acertou miseravi!!!")
        break
    else:
        print("Errou, tente novamente ", end="")
        time.sleep(1)
        print(". ", end="")
        time.sleep(1)
        print(". ", end="")
        time.sleep(1)
        print(". ")
        time.sleep(1)

print(f"Isso, PARABÉNS, o número é {usuario}!")
print(f"Você precisou de {tentativas} chances para adivinhar.")

