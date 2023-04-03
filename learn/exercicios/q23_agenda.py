"""
Desenvolva uma aplicação que simule uma Agenda Pessoal com capacidade para até 10 contatos armazenados. O usuário deve ser capaz de:
cadastrar o nome, o telefone e o email de um novo contato
excluir um contato existente
buscar um contato pelo nome
listar os contatos cadastrados
sair da aplicação
O programa deve ser capaz de verificar se a agenda está cheia e só permitir um novo cadastro caso não tenha atingido a capacidade.
"""

def buscar_contato(lista):
    x = input("Informe o nome do contato a ser buscado: ")
    pos_x = 0
    for i in range(len(lista)):
        if lista[i]['nome'] == x:
            pos_x = i
            break
    print(f"Contato: {lista[pos_x]}")
    return pos_x

while True:
    opcao = int(input("""Escolha uma ação:
    [1] Cadastrar contato
    [2] Buscar contato 
    [3] Excluir contato
    [4] Listar contatos
    [0] Sair 
    opção: """))
    lista = []
    if opcao == 1:
        # cadastrar
        for i in range(10):
            contato = {}
            contato['nome'] = input("Informe o nome do contato: ")
            contato['fone'] = input("Informe o telefone do contato: ")
            contato['email'] = input("Informe o email do contato: ")
            lista.append(contato)

    elif opcao == 2:
        # buscar
        pos_x = buscar_contato(lista)

    elif opcao == 3:
        # excluir
        pos_x = buscar_contato(lista)
        lista[pos_x] = None

    elif opcao == 4:
        # listar
        for contato in lista:
            print(contato)

    else:
        break








