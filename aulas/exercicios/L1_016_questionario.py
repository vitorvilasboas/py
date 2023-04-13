"""
Desenvolva um programa que leia uma matriz 10x5 em que as linhas representam as respostas de 5 questões de múltipla escolha (A, B, C ou D) dadas por 10 alunos. Leia uma lista de 5 valores correspondentes ao gabarito das questões. O programa deve conter funções para comparar as respostas de cada aluno com o gabarito e emitir uma lista contendo a pontuação correspondente a cada estudante (número de acertos) ordenada por pontuação de forma decrescente.
"""
from random import choice

def compara(resp, gab):
    return len(['' for r,g in zip(resp, gab) if r == g])

def rankear(acertos):
    acertos = sorted(acertos, key=lambda x: x[1], reverse=True)
    print("Ranking por acertos:")
    for i in range(len(acertos)):
        print(f"{i+1}º. {acertos[i][0]} ({acertos[i][1]} acertos)")


if __name__ == '__main__':
    alunos = []
    respostas = []
    for i in range(10):
        alunos.append(input(f"Informe o nome do {i+1}º aluno: ").upper())
        print(f"Informe as 5 respostas dadas por {alunos[-1]} [A,B,C,D]:")
        respostas.append([ input(f"Q{j+1}: ").upper() for j in range(5)])

    Q = ['A', 'B', 'C', 'D']
    gabarito = [choice(Q), choice(Q), choice(Q), choice(Q), choice(Q)]
    print(f"GABARITO: {gabarito}")

    # for resp,aluno in zip(respostas, alunos): print(aluno, resp)

    acertos = [ [aluno, compara(resp, gabarito)] for resp,aluno in zip(respostas, alunos) ]

    rankear(acertos)
