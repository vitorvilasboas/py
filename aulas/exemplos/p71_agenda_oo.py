"""
Crie uma aplicação que simule uma Agenda Pessoal com capacidade para até 10 contatos armazenados. O usuário deve ser capaz de:

- cadastrar o nome, o telefone e o email de um novo contato
- excluir um contato existente
- buscar um contato pelo nome
- listar os contatos cadastrados
- sair da aplicação

O programa deve ser capaz de verificar se a agenda está cheia e só permitir um novo cadastro caso não tenha atingido a capacidade.
"""


class Agenda:
    '''
        Definindo a classe Agenda, que vai ser responsáveis pelos métodos seguintes
    '''

    counter = 0

    def __init__(self):

        '''
            Método construtor que vai inicializar a nossa classe
        '''

        self.__contatos = {}  # Criando o dicionário que irá armazenar o nome, telefone e e-mail da pessoa desejada

        self.Options()  # Indo para o método Options, que vai nos apresentar as opções disponíveis na classe agenda

    def Options(self):

        if (Agenda.counter == 0):
            # Aqui exibimos uma frase que só é mostrada uma única vez, quando o objeto dessa classe é declarada.

            print("\nVocê iniciou uma agenda pessoal com limite de 10 contatos.")
            Agenda.counter += 1

        # Bloco de prints mostrando as opções disponíveis

        print("\n\nSegue as operações possíveis para continuar:\n")
        print("1. Adicionar um novo contato ou sobreescrever um contato pré-existente.")
        print("2. Excluir um contato pelo nome.")
        print("3. Encontrar um contato pelo nome.")
        print("4. Listar todos os contatos.")
        print("5. Sair da aplicação.\n")

        # Solicitando a entrada do usuário para saber para qual método iremos

        self.__Option = input("Qual operação deseja realizar? ")

        if (self.__Option == "1"):
            # Bloco de input para recolher do usuário as variáveis necessárias para poder adicionar um novo contato ou sobreescrevê-lo

            Name = input("\nDigite o nome do contato: ")
            Phone = input("Digite o telefone do contato: ")
            Email = input("Digite o e-mail do contato: ")
            print("\n\n")

            self.ADD(Name, Phone, Email)

        if (self.__Option == "2"):
            # Input para recolher do usuário a variável necessária para poder excluir um contato

            Name = input("\nDigite o nome do contato: ")
            print("\n\n")

            self.EXCL(Name)

        if (self.__Option == "3"):
            # Input para recolher do usuário a variável necessária para poder encontrar os dados de um contato

            Name = input("\nDigite o nome do contato: ")
            print("\n\n")

            self.find(Name)

        if (self.__Option == "4"):
            # Chamando o método que exibe os contatos na ordem em que foram adicionados.

            print("\n\n")
            self.listagem()

    def ADD(self, nome, telefone, email):

        # Método para adicionar um novo contato à agenda.

        sobrescrito = 0

        if (nome in self.__contatos.keys()):
            # Condicional que verifica se o nome já existe e notifica ao usuário que estará acontecendo a sobrescrição do contato

            print("Esse contato já está salvo. O contato foi sobrescrito.")
            sobrescrito = 1

        if ((len(self.__contatos) <= 9) or (sobrescrito == 1)):

            # Condional que limita o número de contatos

            self.__contatos[nome] = telefone + ":" + email
            print("O contato foi adicionado!")

        else:

            print("Desculpe, a agenda está cheia.")

        self.Options()  # Voltando para as opções

    def EXCL(self, nome):

        # Método para excluir um contato da agenda.

        if (nome in self.__contatos.keys()):

            # Condicional que checa se o nome está no dicionário para poder apagá-lo

            self.__contatos.pop(nome)
            print("O contato foi excluído!")

        else:

            print(f"Desculpe, {nome} não está na agenda.")

        self.Options()

    def find(self, nome):

        # Método para encontrar um contato da agenda, trazendo seu telefone e e-mail

        if (nome in self.__contatos.keys()):

            # Condicional que checa se o nome está no dicionário e informa seus dados

            print(
                f"O telefone e o e-mail de {nome} são: {self.__contatos[nome].split(':')[0]} e {self.__contatos[nome].split(':')[1]}, respectivamente.")

        else:

            print(f"Desculpe, {nome} não está na agenda.")

        self.Options()

    def listagem(self):

        # Método para listar todos os contatos na ordem em que foram adicionados

        print("Nome\t\tTelefone\t\tE-mail")

        i = 0
        while (i < len(self.__contatos)):
            # Ciclo while que vai imprimir todos os contatos

            print(
                f"{list(self.__contatos.keys())[i]}\t\t{self.__contatos[list(self.__contatos.keys())[i]].split(':')[0]}\t\t{self.__contatos[list(self.__contatos.keys())[i]].split(':')[1]}")

            i += 1

        self.Options()


Teste = Agenda()