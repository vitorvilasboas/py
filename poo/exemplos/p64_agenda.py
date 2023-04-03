"""
Exemplo. Crie um programa em Python que solicite nome, idade, endereço, telefone e estado civil de uma pessoa e
armazene-os em uma lista. Faça a mesma coisa para mais quatro pessoas e armazene todas as listas criadas em uma lista única. Altere as listas incluindo o CPF de cada pessoa. Ao final mostre os dados de todas as pessoas de forma organizada usando saídas formatada e em seguida mostre apenas os endereços cadastrados.
"""

# nome = input("Informe o nome da 1ª pessoa: ")
# idade = int(input("Informe a idade da 1ª pessoa: "))
# end = input("Informe o endereço da 1ª pessoa: ")
# fone = input("Informe o telefone da 1ª pessoa: ")
# estado_civil = input("Informe o estado civil da 1ª pessoa: \n"
#                      "(1) Casado\n"
#                      "(2) Solteiro\n"
#                      "(3) Viúvo\n"
#                      "(4) Divorciado: ")
# p1 = [nome, idade, end, fone, estado_civil]  # cria lista chamada p1 com os dados lidos
# # p1 = (nome, idade, end, fone, estado_civil)  # cria tupla chamada p1 com os dados lidos (imutável)

# p1 = []
p1 = list()  # cria uma lista vazia chamada p1
p1.append(input("Informe o nome da 1ª pessoa: "))  # adiciona o nome na lista p1
p1.append(int(input("Informe a idade da 1ª pessoa: ")))  # adiciona a idade na lista p1
p1.append(input("Informe o endereço da 1ª pessoa: "))  # adiciona o endereço na lista p1
p1.append(input("Informe o telefone da 1ª pessoa: "))  # adiciona o endereço na lista p1
p1.append(input("Informe o estado civil da 1ª pessoa: \n"
                "(1) Casado\n"
                "(2) Solteiro\n"
                "(3) Viúvo\n"
                "(4) Divorciado: "))  # adiciona o estado civil na lista p1

p2 = list()
p2.append(input("Informe o nome da 2ª pessoa: "))
p2.append(int(input(f"Informe a idade de {p2[0]}: ")))
p2.append(input(f"Informe o endereço de {p2[0]}: "))
p2.append(input(f"Informe o telefone de {p2[0]}: "))
p2.append(input(f"Informe o estado civil de {p2[0]}: \n(1) Casado\n(2) Solteiro\n(3) Viúvo\n(4) Divorciado: "))

p3 = list()
p3.append(input("Informe o nome da 3ª pessoa: "))
p3.append(int(input(f"Informe a idade de {p3[0]}: ")))
p3.append(input(f"Informe o endereço de {p3[0]}: "))
p3.append(input(f"Informe o telefone de {p3[0]}: "))
p3.append(input(f"Informe o estado civil de {p3[0]}: \n(1) Casado\n(2) Solteiro\n(3) Viúvo\n(4) Divorciado: "))

p4 = list()
p4.append(input("Informe o nome da 4ª pessoa: "))
p4.append(int(input(f"Informe a idade de {p4[0]}: ")))
p4.append(input(f"Informe o endereço de {p4[0]}: "))
p3.append(input(f"Informe o telefone de {p3[0]}: "))
p4.append(input(f"Informe o estado civil de {p4[0]}: \n(1) Casado\n(2) Solteiro\n(3) Viúvo\n(4) Divorciado: "))

p5 = list()
p5.append(input("Informe o nome da 5ª pessoa: "))
p5.append(int(input(f"Informe a idade de {p5[0]}: ")))
p5.append(input(f"Informe o endereço de {p5[0]}: "))
p3.append(input(f"Informe o telefone de {p3[0]}: "))
p5.append(input(f"Informe o estado civil de {p5[0]}: \n(1) Casado\n(2) Solteiro\n(3) Viúvo\n(4) Divorciado: "))

pessoas = list()  # cria uma lista vazia chamada pessoas

pessoas.append(p1)  # add os dados da lista p1 na lista pessoas
pessoas.append(p2)
pessoas.append(p3)
pessoas.append(p4)
pessoas.append(p5)

print(pessoas)

p1.append(input(f"Informe o CPF de {p1[0]}: "))  # adiciona um novo dado na lista p1
p2.append(input(f"Informe o CPF de {p2[0]}: "))  # ...
p3.append(input(f"Informe o CPF de {p3[0]}: "))
p4.append(input(f"Informe o CPF de {p4[0]}: "))
p5.append(input(f"Informe o CPF de {p5[0]}: "))  # adiciona um novo dado na lista p5
pessoas[0] = p1  # atualiza a lista de pessoas
pessoas[1] = p2  # ...
pessoas[2] = p3
pessoas[3] = p4
pessoas[4] = p5  # atualiza a lista de pessoas

print(f"""Dados de {pessoas[0][0]}:
          Idade: {pessoas[0][1]}
          Endereço: {pessoas[0][2]}
          Fone: {pessoas[0][3]}
          Estado Civil: {pessoas[0][4]}
          CPF: {pessoas[0][5]}""")

print(f"""Dados de {pessoas[1][0]}:
          Idade: {pessoas[1][1]}
          Endereço: {pessoas[1][2]}
          Fone: {pessoas[1][3]}
          Estado Civil: {pessoas[1][4]}
          CPF: {pessoas[1][5]}""")

print(f"""Dados de {pessoas[2][0]}:
          Idade: {pessoas[2][1]}
          Endereço: {pessoas[2][2]}
          Fone: {pessoas[2][3]}
          Estado Civil: {pessoas[2][4]}
          CPF: {pessoas[2][5]}""")

print(f"""Dados de {pessoas[3][0]}:
          Idade: {pessoas[3][1]}
          Endereço: {pessoas[3][2]}
          Fone: {pessoas[3][3]}
          Estado Civil: {pessoas[3][4]}
          CPF: {pessoas[3][5]}""")

print(f"""Dados de {pessoas[4][0]}:
          Idade: {pessoas[4][1]}
          Endereço: {pessoas[4][2]}
          Fone: {pessoas[4][3]}
          Estado Civil: {pessoas[4][4]}
          CPF: {pessoas[4][5]}""")

print(pessoas[0][2], pessoas[1][2], pessoas[2][2], pessoas[3][2], pessoas[4][2])  # mostra somente os endereços cadastrados