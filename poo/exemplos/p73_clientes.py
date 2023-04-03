class Cliente:
    def __init__(self, nome, email):
        self.nome = nome
        self.email = email

class Produto:
    produtos = []
    imposto = 1.05   # 5% de imposto
    desconto = 0.10  # 10% de desconto
    contador = 0

    def __init__(self, name, desc, mark, price):
        self.__codigo = Produto.contador + 1
        self.nome = name
        self.__descricao = desc
        self.marca = mark
        self.preco = (price * Produto.imposto) - (price * Produto.desconto)
        Produto.contador = self.codigo
        self.cliente = None
        self.fornecedor = None


    @property
    def codigo(self):
        return self.__codigo

    @codigo.setter
    def codigo(self, novo_codigo):
        self.__codigo = novo_codigo

    def cadastrar(self):
        pass

    def excluir(self):
        pass

    def editar(self):
        pass


class Fornecedor:
    def __init__(self, cnpj, nome):
        self.__cnpj = cnpj
        self.__nome = nome
        self.__razao_social = None # atributo privado
        self.__fone = None
        self.__email = None
        self.__endereco = None

    def get_nome(self):
        return self.__nome


class ContaCorrente:

    def __init__(self, numero, limite, saldo):
        self.numero = numero
        self.limite = limite
        self.saldo = saldo


class Servico:
    pass

class Cliente:
    pass

class Pedido:
    pass

class Ferramenta:
    pass


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    # print_hi('PyCharm')
    prod1 = Produto('PlayStation 5', 'Video Game Console', 'Sony', 4599.90)
    # prod2 = Produto('Iphone 13', 'Smartphone', 'Apple', 11099.00)
    # prod3 = Produto('Xbox S', 'Video Game Console', 'Microsoft', 4550.89)
    # prod4 = Produto("Inspiron 15 3000", "Notebook", "Dell", 39999.00)

    forn1 = Fornecedor('14.546.678000-1/34', 'HK Informática')  #
    prod1.fornecedor = forn1



