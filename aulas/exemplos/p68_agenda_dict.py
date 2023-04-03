"""
Exemplo. Crie um programa em Python que solicite nome, idade, endereço, telefone e estado civil de uma pessoa e
armazene-os em uma lista. Faça a mesma coisa para mais quatro pessoas e armazene todas as listas criadas em uma lista única. Altere as listas incluindo o CPF de cada pessoa. Ao final mostre os dados de todas as pessoas de forma organizada usando saídas formatada e em seguida mostre apenas os endereços cadastrados.
"""

def obterDados():
    # escopo local
    n = input("Informe o nome da pessoa: ")
    i = int(input("Informe a idade da pessoa: "))
    p = float(input("Informe o peso da pessoa: "))
    a = float(input("Informe a altura da pessoa: "))
    pessoa = {'nome': n, 'idade': i, 'peso': p, 'altura': a}
    return pessoa  # retorno da função — devolver


def cadastrarContatos(contatos, pessoa):
    contatos.append(pessoa)
    return contatos


def imprimirContatos(contatos):  # sem retorno
    for p in range(len(contatos)):
        print(f"Contato: {contatos[p]['nome']}, {contatos[p]['idade']}, {contatos[p]['peso']}, {contatos[p]['altura']}.")


if __name__ == '__main__':
    contatos = []
    while True:  # repete até encontrar o break
        individuo = obterDados()  # recebe o retorno da função obterDados
        contatos = cadastrarContatos(contatos, individuo)
        r = input("Quer cadastrar outra pessoa? s=sim, n=não ... ")
        if r != 's':
            break

    print("Contatos cadastrados: ")
    imprimirContatos(contatos)


# def mostrarMenu():
# def cadastrar():
# def editar():
# def excluir():
# def listar():

# nomes = []  # lista vazia chamada nomes
# for i in range(0, 5, 1):   # repete 5 vezes
#   n = input("Informe o nome: ")  # solicita o nome e armazena em n
#   if n != 'fulano':  # compara o nome digitado (n) com ´fulano´
#     nomes.append(n)  # armazena se a comparação form True
#   else:
#     print("Opa a condição é falsa, logo não armazena na lista!")
# print(nomes)  # mostra toda a lista
#
#
# pessoas = []
# for i in range(0, 5, 1):
#   p = {}
#   p['nome'] = input(f"Informe o NOME da {i+1}ª pessoa: ")
#   p['dtnasc'] = input(f"Informe a DATA DE NASCIMENTO da {i+1}ª pessoa: ")
#   p['cpf'] = input(f"Informe o CPF da {i+1}ª pessoa: ")
#   p['email'] = input(f"Informe o E-MAIL da {i+1}ª pessoa: ")
#   p['senha'] = input(f"Informe a SENHA da {i+1}ª pessoa: ")
#   if p['senha'] == '12345678':
#     print("A senha informada não é válida! "
#           "Esta pessoa não entrará na lista!")
#   else:
#     pessoas.append(p)
# print(pessoas)