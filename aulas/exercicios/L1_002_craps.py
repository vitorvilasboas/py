"""
Desenvolva um programa que implemente o jogo de Craps no qual o jogador lança um par de dados, obtendo um valor entre 2 e 12. Se, na primeira jogada, o jogador tirar 7 ou 11, ele ganha. Se tirar 2, 3 ou 12, na primeira jogada, isto é chamado de "craps" e ele perde. Se, na primeira jogada, tirar 4, 5, 6, 8, 9 ou 10, esta será sua pontuação e deverá continuar jogando os dados até tirar o número novamente, no entanto, caso tire um 7 antes disso, ele perde.
"""

import time
lista = []


def sorteio():
    import random
    d1 = random.randint(1, 6)
    print(f"\nDado 1 = {d1} .", end='')
    for i in range(2): time.sleep(1); print('.', end='')
    d2 = random.randint(1, 6)
    print(f"\nDado 2 = {d2}")
    s = d1 + d2
    print(f"SOMA = {s}")
    return s

rodada = 1
pontos = 0
print(f"Iniciando o craps.", end='')
for i in range(2): time.sleep(1); print('.', end='')
print(f"\nSorteando 1ª rodada de dados.", end='')
for i in range(2): time.sleep(1); print('.', end='')
soma = sorteio()
if soma in [7, 11]:
    print(f"\nParabéns a soma dos dados foi {soma} e você venceu.")
    pontos += soma
elif soma in [2, 3, 12]:
    print(f"\nOps, a soma dos dados foi {soma} e você perdeu.")
else:
    pontos += soma
    while True:
        rodada += 1
        print(f"\nSorteando {rodada}ª rodada de dados.", end='')
        for i in range(2): time.sleep(1); print('.', end='')
        soma2 = sorteio()
        if soma2 == 7:
            print(f"Ops, a soma dos dados foi {soma} e você perdeu.")
            break
        else:
            pontos += soma2
            if soma == soma2:
                print(f"Parabéns você repetiu a soma dos dados da 1ª rodada e venceu.")
                break

print(f"Pontuação: {pontos}")
