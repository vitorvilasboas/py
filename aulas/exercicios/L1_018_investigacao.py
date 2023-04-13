"""
Desenvolva um programa capaz de entrevistar um número indefinido de pessoas a fim de desvendar um crime. Leia e armazene em uma coleção de dados o nome do entrevistado e suas respostas para 5 perguntas binárias (Sim ou Não):
    a. Telefonou para a vítima?
    b. Esteve no local do crime?
    c. Mora a menos de 1Km da vítima?
    d. Devia dinheiro para a vítima?
    e. Já trabalhou com a vítima?
O programa deve registrar novas entrevistas até que seja fornecido um nome vazio (“ “). Após encerrar as entrevistas, o programa deve emitir (e armazenar) uma classificação sobre a participação de cada entrevistado no crime, com base no número de respostas positivas conforme a tabela abaixo e informar o nome das pessoas com algum grau de suspeição sobre o crime ordenados dos mais suspeitos aos menos suspeitos.
    Nº de respostas positivas   Classificação
        0 ou 1                  Inocente
        2                       Pouco Suspeito
        3 ou 4                  Suspeito
        5                       Muito Suspeito
"""
E = []  # entrevistados
while True:
    nome = input("Informe o nome do entrevistado?")
    if nome == '': break
    pessoa = {}
    pessoa['nome'] = nome
    pessoa['r1'] = input("Telefonou para a vítima?")
    pessoa['r2'] = input("Esteve no local do crime?")
    pessoa['r3'] = input("Mora a menos de 1Km da vítima?")
    pessoa['r4'] = input("Devia dinheiro para a vítima?")
    pessoa['r5'] = input("Já trabalhou com a vítima?")
    E.append(pessoa)

for i in range(len(E)):
    cont = 0
    for i in range(1,6):
        if E[i][f'resp{i}'].upper() == 'S': cont += 1
    E[i]['nota'] = cont

for pessoa in sorted(E, key=lambda p:p['nota'], reverse=True):
    if cont == 2: print(f"{pessoa['nome']} é pouco suspeito")
    elif cont in [3, 4]: print(f"{pessoa['nome']} suspeito")
    elif cont == 2: print(f"{pessoa['nome']} é muito suspeito")