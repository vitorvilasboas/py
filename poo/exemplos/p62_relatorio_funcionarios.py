# -*- coding: utf-8 -*-
"""
Crie um programa que leia o nome, o cargo, a idade, o sexo e o salário de vários funcionários de uma vitor_poo.
A cada funcionário cadastrado, o programa deverá perguntar se o usuário quer ou não continuar. No final, mostre:
* quantos funcionários com salário acima de R\$2000,00 foram cadastrados.
* os nomes dos funcionários menores de 18 anos.
* qual a média de salário dos funcionários homens.
* qual a média de salário das funcionárias mulheres com mais de 30 anos de idade.
"""

funcionarios = []
while (True):
    op = int(input(f'''
                    ==== Cadastro de funcionários ====
                    [1] Cadastrar novo funcionário
                    [2] Sair e gerar relatório
                    {'=' * 34}
                    opção: '''))
    print("\n")
    if op == 1:
        print('=== Novo Funcionário ===')
        nome = str(input("Nome: "))
        cargo = str(input("Cargo: "))
        idade = int(input("Idade: "))
        sexo = str(input("Sexo [M/F]: ")).upper()
        salario = float(input("Salario: R$"))
        func = {'nome': nome, 'cargo': cargo, 'idade': idade, 'sexo': sexo, 'salario': salario}
        funcionarios.append(func)
    elif op == 2:
        print("Saindo...")
        break
    else:
        print('OPÇÃO INVÁLIDA!')

print("\n")
print("=== Relatório ===")

if funcionarios == []:
    print("Nenhum funcionário cadastrado.")
else:
    # quantos funcionários com salário acima de R$2000,00 foram cadastrados.
    cont = 0
    for f in funcionarios:
        if f['salario'] > 2000: cont += 1
    print(f" {cont} funcionários têm salário acima de R$2000,00.")

    # nomes dos funcionários menores de 18 anos.
    print(f" {cont} Funcionários menores de 18 anos:")
    for f in funcionarios:
        if f['idade'] < 18: print(f['nome'])

    # média de salário dos funcionários homens e
    # média de salário das funcionárias mulheres com mais de 30 anos de idade.
    soma_sal_h, cont_h = 0, 0
    soma_sal_m30, cont_m30 = 0, 0
    for f in funcionarios:
        if f['sexo'] == 'M':
            soma_sal_h += f['salario']
            cont_h += 1
        else:
            if f['salario'] > 30:
                soma_sal_m30 += f['salario']
                cont_m30 += 1

    print(f" Média de salário entre os funcionários do sexo masculno: R${(soma_sal_h / cont_h):2.f}")
    print(f" Média de salário entre os funcionárias mulheres maiores que 30 anos: R${(soma_sal_m30 / cont_m30):2.f}")
