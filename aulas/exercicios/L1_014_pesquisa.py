"""
Desenvolva um programa que simule uma pesquisa entre os habitantes de uma região solicitando o nome, a idade, o sexo e o salário de uma quantidade, à princípio, indeterminada de entrevistados. A entrada de dados deve ser encerrada quando for digitado “fim” para o campo nome ou uma idade negativa. Ao final, o programa deve apresentar um relatório com:
    a. a média de salário do grupo;
    b. quantos homens e quantas mulheres há no grupo
    c. os dados das pessoas com maior e menor idade do grupo;
    d. a quantidade de mulheres com salário até R$100,00.
    e. quantos homens possuem mais de 20 anos.
"""

import numpy as np

pessoas = []
print("==== Pesquisa Social ====")
while True: # for i in range(10):
    nome = input("\nNome do entrevistado ('fim' para encerrar): ")
    if nome.lower() == 'fim': break
    idade = int(input("Idade do entrevistado (negativo para encerrar): "))
    if idade < 0: break
    p = {'nome': nome, 'idade': idade}
    p['sexo'] = input("Sexo do entrevistado [M]asculino ou [F]eminino: ").upper()
    p['salario'] = float(input("Salário do entrevistado: R$"))
    pessoas.append(p)

if 'p' in locals(): # verifica se o dicionário p existe no escopo local
    print(f"\nMédia de salários: {np.mean(np.array([p['salario'] for p in pessoas]))}")
    print(f"Quantidade Homens: {len([p for p in pessoas if p['sexo'] == 'M'])}")
    print(f"Quantidade Mulheres: {len([p for p in pessoas if p['sexo'] == 'F'])}")
    print(f"Mais novo: {sorted(pessoas, key=lambda p: p['idade'])[0]}")
    print(f"Mais velho: {sorted(pessoas, key=lambda p: p['idade'])[-1]}")
    print(f"Qtd Mulheres c/ salário > R$100,00: {len([p for p in pessoas if p['sexo'] == 'F' and p['salario'] > 100])}")
    print(f"Qtd Mulheres c/ mais de 20 anos: {len([p for p in pessoas if p['sexo'] == 'M' and p['idade'] > 20])}")