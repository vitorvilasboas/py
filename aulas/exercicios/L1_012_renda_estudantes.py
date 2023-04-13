"""
Desenvolva um programa que leia o nome, a matrícula, a data de nascimento e a renda familiar de 10 estudantes. Calcule e armazene na mesma coleção de dados as idades de cada estudante. Ao final, determine quantos dos estudantes cadastrados têm idade menor que 18 anos e possuem renda familiar inferior a R$2000,00.
"""
from datetime import datetime
alunos = []
while True: # for i in range(10):
    if input("Novo estudante? [S]im ou [N]ão: ").upper() != 'S': break
    aluno = {'nome': input("Informe o nome do novo estudante: ")}
    aluno['matrícula'] = input(f"Matrícula de {aluno['nome']}: ")
    aluno['nascimento'] = datetime.strptime(input(f"Data de nascimento de {aluno['nome']} (dd/mm/aaaa): "), '%d/%m/%Y')
    aluno['renda'] = float(input(f"Renda familiar de {aluno['nome']}: R$"))
    aluno['idade'] = datetime.today().year - aluno['nascimento'].year
    alunos.append(aluno)

cont = 0
lista = []
for aluno in alunos:
    if aluno['idade'] < 18 and aluno['renda'] < 2000:
        cont += 1
        lista.append(aluno)

print(f'{cont} estudantes menores de idade com renda familiar inferior à R$2000,00, são eles:')
for aluno in lista: print('{} ({}) >> idade: {} anos, renda: {:.2f}'.format(aluno['nome'], aluno['matricula'], aluno['idade'], aluno['renda']))

