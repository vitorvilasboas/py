"""
Desenvolva um programa que simule um jogo de loteria. Crie uma função para sortear 6 números inteiros distintos entre 1 e 60 e armazene-os em uma lista. Leia quantas apostas o usuário desejar, armazenando-as numa coleção de dados única. Cada aposta deve conter um código de 5 dígitos gerado automaticamente, o CPF do apostador e os seis números de sua escolha. Após coletar todas as apostas, utilize uma função para comparar os números de cada aposta com os números sorteados. Ao final, mostre os dados das apostas na ordem decrescente do número de acertos.
"""
from random import randint, sample
from time import sleep
from os import system

def sortear():
    # return [ randint(1, 60) for i in range(6) ] # cria lista com 6 números aleatórios (possível repetir)
    return sample(list(range(1, 61)), 6) # cria lista com 6 números aleatórios sem repetições


if __name__ == '__main__':
    cod = randint(1000, 9000)
    apostas = []
    while True:
        if input("Registrar nova aposta? [S]im [N]ão: ").upper() != 'S':
            break
        system("cls")
        cod += 1
        print(f"Código da aposta: {cod}")
        aposta = {}
        aposta['codigo'] = cod
        aposta['cpf'] = input("Informe o CPF do apostador: ")
        aposta['numeros'] = input("Informe os 6 números da aposta (entre 1 e 60) separados por vírgula: ")
        aposta['numeros'] = aposta['numeros'].strip().split(',')
        aposta['numeros'] = list(map(int, aposta['numeros']))  # converter lista p/ int
        apostas.append(aposta)

    sleep(2)
    print(f"Realizando sorteio.", end='')
    for i in range(2): sleep(1); print('.', end='')
    sorteio = sortear()
    print(f"\nNúmeros sorteados: {sorteio}")

    for aposta in apostas:
        cont = 0
        for x in aposta['numeros']:
            if sorteio.count(x) > 0: cont += 1
        aposta['acertos'] = cont

    sleep(2)
    print(f"Analisando apostas.", end='')
    for i in range(2): sleep(1.5); print('.', end='')
    print("\nApostas ganhadoras: ")

    ganhadores = []
    for aposta in apostas:
        if aposta['numeros'] == sorteio:
            ganhadores.append(aposta)
            print(f"Aposta nº {aposta['codigo']} - CPF: {aposta['cpf']}")

    func = lambda aposta: aposta['acertos']
    apostas = sorted(apostas, key=func, reverse=True)

    print("\nOrdem de apostas por acerto: ")
    for aposta in apostas: print(f"Aposta nº {aposta['codigo']} - CPF: {aposta['cpf']} (acertos: {aposta['acertos']})")
