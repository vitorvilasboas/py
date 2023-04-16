# ===============
# LISTAS
# ===============
# carros = []   # lista vazia
# carros = list()  # lista vazia
# carros = ['Pálio', 'Gol', 'Cobalt', 'Uno', 'Versa', 'Up', 'Corolla', 'SW4', 'Golf', 'Hilux']
# len(carros)  # tamanho
# carros.append('S10') # add elemento no fim da lista
# carros.insert(3,'Civic') # add elemento na posição específica
# del carros[1]   # exclusão pelo indice (modo 1)
# carros.pop(2)   # exclusão pelo indice (modo 2)
# carros.pop()    # exclusão do último elemento (pop sem argumento)
# carros.remove('Up')     # exclusão pelo valor
# carros.index('Versa')   # índice do elemento
# carros.count('Gol')   # conta elemento
# carros.sort()   # ordena asc
# carros.sort(reverse=True)   #ordena desc
# sorted(carros)   # ordena asc (modo 2)
# sorted(carros, reverse=True)   # ordena desc (modo 2)
# veiculos = carros  # relação (interdependência)
# veiculos = carros[:]  # cópia via fatiamento completo
# veiculos = carros.copy()  # cópia via copy()
# carros.clear()   # limpa lista

# numeros = list(range(20))  # lista de um intervalo
# numeros = list(range(1, 20))
# numeros = list(range(20, 1, -1))
# max(numeros)   # maior
# min(numeros)   # menor

# [ i**2 for i in range (1, 11)]  # List comprehentions

# enumerate - enumera itens da lista retorna indice e item
# for i, item in enumerate(carros):
#     print(f'A posição é {pos} e o item é {item}')

# zip - itera varios indices ao mesmo tempo
# for i, j in zip(range(0, 5), range(5, 10)):
#     print(i, '>>>', j)

# carros = [][]
# carros = [ ['Cobalt', 'GM', 2013], ['Cronos', 'FIAT', 2018], ['Corolla', 'Toyota', 2017], ['Ranger', 'Ford', 2020] ]
# carros[1]
# carros[1][0]
# carros.append(['Gol', 'Volkswagem', 2002])
# carros.insert(1, ['Palio', 'Fiat', 2016])  # inserir na posição

# for i in range(len(carros)):
#   for j in range(len(carros[0])):
#       print(carros[i][j])

# pessoa = [1.85, 35, 'Vitor', [12, 2, 1988]]

"""
Exemplo: Leia o salário de cinco funcionários e calcule o desconto do imposto de seguridade social de cada um deles sabendo que para salários superiores a R$5000 o desconto é de 22% e para salários até R$5000 o desconto é de 10%. Mostre o salário mais alto após o desconto.
"""
# salarios = []
# for i in range(0, 5):
#   salario = float(input(f"Entre com o salário do {i + 1}o funcionário: R$"))
#   salario -= (salario * 0.22) if salario > 5000 else (salario * 0.10)
#   salarios.append(salario)
#   print(f"==> Salário ajustado: R${salario:.2f}\n")
# print(f"==> Maior salário ajustado: R${max(salarios):.2f}")

"""
Exemplo: Informe a idade dada data de nascimento
"""
# import datetime
# dt_nascimento = int(input("Informe sua data de nascimento: ").split('/'))
# dia, mes, ano = dt_nascimento[0], dt_nascimento[1], dt_nascimento[2]
# ano_atual = datetime.date.today().year
# idade = ano_atual - ano

"""
Exemplo: Um professor quer sortear a ordem de apresentação do trabalho dos lista1_alunos. Leia o nome de cinco lista1_alunos, armazene-os em uma lista e mostre a ordem sorteada.
"""
# from random import randint
# from time import sleep
# print(f"Estudante sorteado: \033[1m{['MARCELO', 'MARIA','NICOLAS','PEDRO','ESTER'][randint(0, 4)]}\033[0m")
# # OU
# lista1_alunos = ['MARCELO', 'MARIA','NICOLAS','PEDRO','ESTER']
# n = randint(0, len(lista1_alunos)-1)
# sleep(2)
# print(f"Estudante sorteado: \033[1m{lista1_alunos[n]}\033[0m")

"""
Exemplo: Solicite o nome, a idade e a cidade de 5 usuários e armazene todos esses dados primeiro em uma lista para cada usuário e depois em uma lista única. Ao final mostre:
* apenas as idades maiores de 18 anos.
* os nomes dos usuários com idade menor que 18 anos.
"""
# pessoas = list()
# for i in range(5):
#   p = list()
#   p.append(input("Informe seu nome: "))
#   p.append(int(input("Informe sua idade: ")))
#   p.append(input("Informe sua cidade: "))
#   pessoas.append(p)
#
# for i in range(5):
#   print(pessoas[i])
#
# for i in range(5):
#   if pessoas[i][1] >= 18:
#     print(pessoas[i][1], end=" ")
#   else:
#     print(pessoas[i][0], end=" ")
