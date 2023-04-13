"""
Desenvolva um programa que faça o computador "pensar" em um número inteiro entre 0 e 10 e peça para o usuário tentar descobrir qual foi o número escolhido pelo computador. O programa deverá escrever na tela se o usuário acertou ou errou o número e caso tenha errado deve solicitar outro número até ele acertar. Ao final, mostre quantos palpites foram necessários até o acerto.
"""
from random import randint
from time import sleep

print("Pensando em número de 1 a 10.", end='')
for i in range(2): sleep(1); print('.', end='')
comput = randint(1,10)

usuario = int(input("\nPensei, tente adivinhar qual foi: "))
tentativas = 1
while usuario != comput:
    for i in range(3): sleep(1); print('. ', end='')
    usuario = int(input("\nERRADO, tente novamente: "))
    tentativas += 1
for i in range(3): sleep(1); print('.', end='')
print(f"\nIsso, PARABÉNS, o número é {comput}!")
print(f"Você precisou de {tentativas} tentativas para adivinhar.")