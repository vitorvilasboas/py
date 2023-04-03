"""
POO - Métodos

- Métodos (funções) -> Representam os comportamentos do objeto. Ou seja, as ações
que este objeto pode realizar no seu sistema.

Em Python, dividimos os métodos, em 2 grupos: Métodos de instância
e Métodos de Classe.

# Métodos de Instância

# O método dunder init __init__ é um método especial chamado de construtor e
sua função é construir o objeto a partir da classe.

OBS: Todo elemento em Python que inicia e finaliza com duplo underline é chamado de dunder (Double Underline)

OBS: Os métodos/funções dunder em Python são chamados de métodos mágicos.

ATENÇÃO! Por mais que possamos criar nossas próprias funções utilizando dunder (underline no início e no fim)
não é aconselhado. Python possui vários métodos com esta forma de nomenclatura e pode ser que mudemos o comportamento
dessas funções mágicas internas da linguagem. Então, evite ao máximo. De preferência nunca o faça.

# Métodos são escritos em letras minúsculas. Se o nome for composto, o nome terá as palavras separadas por underline.

p1 = Produto('Playstation 4', 'Video Game', 2300)

print(p1.desconto(50))


print(Produto.desconto(p1, 40))  # self, desconto

user1 = Usuario('Angelina', 'Jolie', 'angelina@gmail.com', '123456')
user2 = Usuario('Felicity', 'Jones', 'felicity@gmail.com', '654321')

print(user1.nome_completo())

print(Usuario.nome_completo(user1))

print(user2.nome_completo())

print(f'Senha User 1: {user1._Usuario__senha}')  # Acesso de forma errada de um atributo de classe
print(f'Senha User 2: {user2._Usuario__senha}')  # Acesso de forma errada de um atributo de classe

nome = input('Informe o nome: ')
sobrenome = input('Informe o sobrenome: ')
email = input('Informe o e-mail: ')
senha = input('Informe a senha: ')
confirma_senha = input('Confirme a senha: ')

if senha == confirma_senha:
    user = Usuario(nome, sobrenome, email, senha)
else:
    print('Senha não confere...')
    exit(1)

print('Usuário criado com sucesso!')

senha = input('Informe a senha para acesso: ')

if user.checa_senha(senha):
    print('Acesso permitido')
else:
    print('Acesso negado')

print(f'Senha User Criptografada: {user._Usuario__senha}')  # Acesso errado

# Métodos de Classe em Python são conhecidos como Métodos Estáticos em outras linguagens.

# Métodos de Classe

user = Usuario('Felicity', 'Jones', 'felicity@gmail.com', '123456')

Usuario.conta_usuarios()  # Forma correta
user.conta_usuarios()  # Possível, mas incorreta

user = Usuario('Felicity', 'Jones', 'felicity@gmail.com', '123456')

print(user._Usuario__gera_usuario())  # Acesso, de forma ruim...
"""


class Lampada:

    def __init__(self, cor, voltagem, luminosidade):
        self.__cor = cor
        self.__voltagem = voltagem
        self.__luminosidade = luminosidade
        self.__ligada = False


class ContaCorrente:

    contador = 4999

    def __init__(self, limite, saldo):
        self.__numero = ContaCorrente.contador + 1
        self.__limite = limite
        self.__saldo = saldo
        ContaCorrente.contador = self.__numero


class Produto:

    contador = 0

    def __init__(self, nome, descricao, valor):
        self.__id = Produto.contador + 1
        self.__nome = nome
        self.__descricao = descricao
        self.__valor = valor
        Produto.contador = self.__id

    def desconto(self, porcentagem):
        """ Retorna o valor do produto com o desconto """
        return (self.__valor * (100 - porcentagem)) / 100


from passlib.hash import pbkdf2_sha256 as cryp


class Usuario:

    contador = 0

    @classmethod
    def conta_usuarios(cls):
        print(f'Classe: {cls}')
        print(f'Temos {cls.contador} usuário(s) no sistema')

    @classmethod
    def ver(cls):
        print('Teste')

    @staticmethod
    def definicao():
        return 'UXR344'

    def __init__(self, nome, sobrenome, email, senha):
        self.__id = Usuario.contador + 1
        self.__nome = nome
        self.__sobrenome = sobrenome
        self.__email = email
        self.__senha = cryp.hash(senha, rounds=200000, salt_size=16)
        Usuario.contador = self.__id
        print(f'Usuário criado: {self.__gera_usuario()}')

    def nome_completo(self):
        return f'{self.__nome} {self.__sobrenome}'

    def checa_senha(self, senha):
        if cryp.verify(senha, self.__senha):
            return True
        return False

    def __gera_usuario(self):
        return self.__email.split('@')[0]




# Método Estático


print(Usuario.contador)

print(Usuario.definicao())

user = Usuario('Felicity', 'Jones', 'felicity@gmail.com', '123456')

print(user.contador)

print(user.definicao())

