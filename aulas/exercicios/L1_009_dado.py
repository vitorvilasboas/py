"""
Desenvolva um programa que simule o lançamento de um dado 100 vezes e armazene os números sorteados aleatoriamente em uma lista. Ao final, mostre quantas vezes cada um dos números foi sorteado (de 1 até 6).
"""
from random import randint
lista = [ randint(1, 6) for i in range(100) ]
print(f"\nSequência sorteada: {lista}")
for i in range(1, 7): print(f"Número {i} foi sorteado {lista.count(i)} vezes.")

# from random import randint
# from time import sleep
# import os
#
# os.system("cls")
# sleep(2)
#
# print(f"Jogando o dado.", end='')
# for i in range(2): sleep(1); print('.', end='')
# print("\n")
# lista = []
# for i in range(100):
#     os.system("cls")
#     print(f"Jogando o dado... ({i+1})")
#     sleep(.3)
#     lista.append(randint(1, 6))
#
# sleep(1)
# print(f"\nSequência sorteada:\n{lista}")
#
# print(f"\nCalculando.", end='')
# for i in range(2): sleep(1); print('.', end='')
# print('\n')
# for i in range(1, 7):
#     sleep(1)
#     print(f"Número {i} foi sorteado {lista.count(i)} vezes.")