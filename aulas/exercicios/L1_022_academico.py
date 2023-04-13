"""
Desenvolva um programa que simule um sistema acadêmico cuja interface inicial corresponda a um menu com as seguintes opções:
    1 - Cadastrar aluno
    2 - Buscar aluno
    3 - Emitir relatório completo
    0 - Sair
Na opção 0, o programa deverá ser encerrado após a confirmação do usuário do tipo “Tem certeza que deseja sair? S/N”; Na opção 1, o programa deve ler e armazenar em uma coleção de dados o nome, a matrícula, a série, a turma e as quatro notas bimestrais de um aluno, além de calcular e armazenar a média ponderada das notas de cada aluno de acordo com a fórmula: MP = (nota1*3 + nota2*2 + nota3*3 + nota4*2)/10. Ainda na opção 1 e a partir da média calculada, o programa deve verificar e armazenar o conceito e a situação do aluno conforme tabela abaixo:
    Nota        Conceito    Situação
    0,0 a 3,99  D           Reprovado
    4,0 a 5,99  C           Recuperação
    6,0 a 7,99  B           Aprovado
    8,0 a 10,0  A           Aprovado com mérito
Na opção 2, o programa deve permitir que o usuário busque por um aluno através de sua matrícula e mostrar todos os dados desse aluno caso o encontre na coleção de dados; Na opção 3, o programa deve mostrar um relatório completo que inclui mostrando:
    a. A quantidade e a porcentagem de lista1_alunos classificados em cada conceito;
    b. A quantidade e a porcentagem de lista1_alunos não aprovados;
    c. A média de notas da turma (média das MPs de todos os lista1_alunos);
    d. A média de notas dos lista1_alunos não aprovados;
    e. Os dados do aluno com maior MP;
    f. Os dados do aluno com menor MP.
"""

from time import sleep

def menu():
    print("\n=== SISTEMA ACADÊMICO ===")
    while True:
        sleep(2)
        op = input("O que quer fazer?\n[1] Cadastrar aluno\n[2] Buscar aluno\n[3] Emitir relatório\n[0] Sair")
        if op in ['0','1','2','3']: break
        else: print("Opção inválida! Tente novamente...")
    return int(op)

def cadastrar(lista):
    # nome, a matrícula, a série, a turma e as quatro notas bimestrais de um aluno
    aluno = {}
    aluno['nome'] = input("Informe o nome do aluno: ")
    return lista

def calcular_media():
    pass
    bdhkfhskdhfksdhfhsdkfh

def definir_conceito():
    pass

def buscar():
    pass

def emitir_relatorio():
    pass

if __name__ == '__main__':
    alunos = list()
    while True:
        op = menu()
        if op == 0:
            if input("Tem certeza que deseja sair S/N?").upper() != 'N': break
        elif op == 1:
            cadastrar(alunos)
        elif op == 2:
            buscar()
        else: pass

