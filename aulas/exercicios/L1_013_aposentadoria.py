"""
Desenvolva um programa que leia e armazene o nome, o sexo e os anos de carteira assinada de 10 pessoas em uma lista. Calcule e mostre a média entre os anos de carteira assinada, bem como os nomes do homem e da mulher com maiores tempos de carteira assinada. Além disso, o programa deve dizer quantas pessoas estão aptas a se aposentar sabendo que, para tal, os homens devem ter pelo menos 40 anos de carteira assinada e as mulheres pelo menos 35 anos.
"""
import numpy as np
pessoas = []
while True: # for i in range(10):
    if input("Novo cadastro? [S]im ou [N]ão: ").upper() != 'S': break
    p = {'nome': input("Nome: ")}
    p['sexo'] = input("Sexo [M]asculino ou [F]eminino: ").upper()
    p['tempo_carteira'] = int(input("Tempo de carteira assinada (anos): "))
    pessoas.append(p)

if 'p' in locals(): # verifica se o dicionário p existe no escopo local
    tempos = [p['tempo_carteira'] for p in pessoas]
    media = np.mean(np.array(tempos))
    print(f'Média de tempos de carteira assinada: {media}')

    homens = [p for p in pessoas if p['sexo'] == 'M']
    mulheres = [p for p in pessoas if p['sexo'] == 'F']
    maxM = sorted(homens, key=lambda h: h['tempo_carteira'])[-1]
    maxF = sorted(mulheres, key=lambda m: m['tempo_carteira'])[-1]
    print(f"Homem com maior tempo de carteira assinada: {maxM['nome']} ({maxM['tempo_carteira']} anos)")
    print(f"Mulher com maior tempo de carteira assinada: {maxF['nome']} ({maxF['tempo_carteira']} anos)")

    cont = 0
    aptos = []
    for p in pessoas:
        if (p['sexo'] == 'M' and p['tempo_carteira'] > 40) or (p['sexo'] == 'F' and p['tempo_carteira'] > 35):
            cont += 1
            aptos.append(p)

    print(f'{cont} pessoas aptas a se aposentar, são elas:')
    for p in aptos: print(f"{p['nome']} >> Sexo: {p['sexo']}, Tempo de carteira: {p['tempo_carteira']}")
