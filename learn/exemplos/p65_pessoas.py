"""
Crie um programa que solicite o nome, a idade e o endereço de algumas
pessoas e armazene todos esses dados em uma lista de contatos.
"""

###### for — exige controle do num de repetições
qtd_pessoas = int(input("Quantos pessoas você quer adicionar? "))
contatos = []
for contador in range(0, qtd_pessoas):
    dicionario = {}
    dicionario['nome'] = input(f"Informe o nome da {contador+1}ª pessoa: ")
    dicionario['idade'] = int(input(f"Informe a idade da {contador+1}ª pessoa: "))
    dicionario['endereco'] = input(f"Informe o endereço da {contador+1}ª pessoa: ")
    contatos.append(dicionario)
    print("\n")
print(contatos)


###### while
contatos = []
resposta = "s"
contador = 0
while resposta == 's':
    dicionario = {}
    dicionario['nome'] = input(f"Informe o nome da {contador+1}ª pessoa: ")
    dicionario['idade'] = int(input(f"Informe a idade da {contador+1}ª pessoa: "))
    dicionario['endereco'] = input(f"Informe o endereço da {contador+1}ª pessoa: ")
    contatos.append(dicionario)
    print("\n")
    contador = contador + 1
    resposta = input("Dados salvos. Quer cadastrar outro contato? s-Sim n-Não ")
print(contatos)


###### do ... while
contatos = []
contador = 0
while True:
    dicionario = {}
    dicionario['nome'] = input(f"Informe o nome da {contador+1}ª pessoa: ")
    dicionario['idade'] = int(input(f"Informe a idade da {contador+1}ª pessoa: "))
    dicionario['endereco'] = input(f"Informe o endereço da {contador+1}ª pessoa: ")
    contatos.append(dicionario)
    print("\n")
    contador = contador + 1
    resposta = input("Dados salvos. Quer cadastrar outro contato? s-Sim n-Não ")
    if resposta == 'n':
        break
print(contatos)



