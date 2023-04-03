"""
POO — elementos essenciais:
Classe: Modelo computacional para um conjunto de objetos similares no contexto de um problema real.
Objeto: Instância (exemplo) de uma classe.
Construtor: Método especial utilizado para criar (instanciar) os objetos;
Atributo: Características/propriedades do objeto que definem seu estado. Podem ser: de classe ou de instância
Métodos: Definem os comportamentos/funções/ações executadas por objetos de uma classe
"""

# Para definir uma classe utilizamos a palavra reservada class
# Por convenção, nome de classe deve inicar com maiúscula (padrão CammelCase p/ nome composto)
class Usuario:
    """ Classe que abstrai um usuário cadastrado no sistema """
    # dentro de uma classe tenho atributos/métodos
    # Atributos de classe — valores estáticos para todos objetos; geralmente já inicializados com um valor default (padrão)
    dominio = '@ifto.edu.br'
    contador = 0

    def __init__(self, name, username, password):  # método construtor da classe
        # self é convenção para 'este objeto instanciado' (lembra restaurante self-service)
        # Atributos de instância — declarados dentro do construtor; valores individualizados por objeto
        self.codigo = Usuario.contador + 1
        self.nome = name    # atributo público (visível/acessível em todoo projeto)
        self.email = username + self.dominio
        self.__senha = password   # atributo privado (visível/acessível só na classe)
        Usuario.contador = self.codigo

    def mostra_senha(self):
        print(self.__senha)


# Constrói objetos (instâncias da classe Usuário) — invoca o método construtor da classe
user1 = Usuario("Vitor", "vitorvilasboas", "123456")
user2 = Usuario("Josi", "josivilasboas", "987654")

print(user1)
print(user2)

# Acesso a atributos públicos de instância de um objeto
print(user1.nome, user1.email)  # usa-se o '.' (ponto) entre o objeto e o atributo
print(user2.nome, user2.email)

# Acesso a atributos PRIVADOS de instância de um objeto
print(user1.__senha)  # AttributeError

user1.mostra_senha()

# Acesso a atributos de classe de um objeto
print(user1.dominio)   # não recomendada
print(user2.dominio)
print(Usuario.dominio)  # diretamente pelo nome da classe — RECOMENDADO

user1.nome = "Vitor Vilas Boas"  # Altera o estado do atributo nome do objeto

print(type(user1))      # Objetos são como variáveis do tipo da classe

print(user1.__dict__)   # propriedade __dict__ retorna estado do objeto como dicionário

user2.dt_nasc = '12/02/1988'  # Cria atributo de instância DINÂMICO (em tempo de execução) — exclusivo do objeto que o criou

print(user1.__dict__)   # na classe não existe o atributo dt_nasc, apenas no objeto user2
print(user2.__dict__)

print(f'Usuario: {user2.nome}, Email: {user2.email}, Senha: {user2.senha}, Data Nascimento: {user2.dt_nasc}')

del user2.dt_nasc  # exclui atributo

print(Usuario.__doc__)  # acessao à documentação da Classe Usuario via atributo especial __doc__

print(help(Usuario))  # acesso à documentação da Classe Usuario via funcao help() — descrição completa

print(type(int))  # Módulos Builtins como Tipos de Dados são Classes


"""
Atributos também podem ser:
 * Públicos (pode ser acessado no projeto completo)
 * Privados (pode ser acessado apenas por métodos dentro da própria classe):

Em Java, há palavras chave para definir se um parâmetro é público ou privado:

public class Lampada(){

    public Boolean ligada = false;
    private int voltagem = 110;

    public Lampada(String cor){
        private this.cor = cor;
    }
}

Em Python, por padrão, todos os atributos de uma classe são públicos.
Por convenção, para definir um atributo como privado utiliza-se duplo underscore (__) no início de seu nome.

class Lampada:

    self.ligada = False
    self.__voltagem = 110

    def __init__(self, cor):
        self.__cor = cor


Exemplo:
Imagine que você queira fazer um sistema para automatizar o controle das lâmpadas da sua casa. Atributos no caso da lâmpada: possivelmente iríamos querer saber se a lâmpada é 110 ou 220 volts, se ela é branca, amarela, vermelha ou outra cor, qual é a luminosidade dela e etc (atributos). Provavelmente iríamos querer programar funções básicas da lâmpada como ligar e desligar (métodos).
"""


class Lampada:
    ligada = False  # público
    __voltagem = 110  # privado

    def __init__(self, cor, clareza):
        self.__cor = cor  # privado
        self.copia_cor = self.__cor  # público
        self.luminosidade = clareza
        self.outro = Lampada.__voltagem


# print(Lampada.ligada)
# print(Lampada.__voltagem)

lamp = Lampada("Azul", "80%")

print(lamp.ligada)
print(lamp.outro)

luz = Lampada("Azul", "50%")
# print(luz.cor)
print(luz.copia_cor)

"""
Observações sobre os MÉTODOS:
- Assim como os atributos, os MÉTODOS em uma classe também podem ser de Instância e de Classe (estáticos).
- Os métodos/funções dunder em Python são chamados de *métodos mágicos*.
- Por convenção (PEP8), métodos são escritos em letras minúsculas. Se o nome for composto, o nome terá as palavras separadas por underline.
"""

class Acesso:

    def alterar_senha(self, nova_senha):
        self.__senha = nova_senha

    def __init__(self, email, senha):
        self.email = email
        self.__senha = senha

    def mostra_senha(self):
        print(self.__senha)

    def mostra_email(self):
        print(self.email)


user1 = Acesso('user1@gmail.com', '123456')
user2 = Acesso('user2@gmail.com', '654321')

user1.__dict__

user1.mostra_email()
user2.mostra_email()

"""
Lembre-se: **Métodos de instância** necessitam de uma instânca de classe para ser invocado/utilizado. 
Métodos de classe (estáticos) podem ser invocados diretamente pelo nome da classe (semelhante aos atributos).
"""


class Produto:
    contador = 0

    def __init__(self, nome, descricao, valor):
        self.__id = Produto.contador + 1
        self.__nome = nome
        self.__descricao = descricao
        self.__valor = valor
        Produto.contador = self.__id

    def desconto(self, porcentagem):  # método de instância
        """Retorna o valor do produto com o desconto"""
        return (self.__valor * (100 - porcentagem)) / 100


p1 = Produto('Playstation 4', 'Video Game', 2300)

# chamada método de instância
print(p1.desconto(50))

print(Produto.desconto(p1, 40))  # self, desconto


class Usuario:

    def __init__(self, nome, sobrenome, email, senha):
        self.__nome = nome
        self.__sobrenome = sobrenome
        self.__email = email
        self.__senha = senha

    def nome_completo(self):
        return f'{self.__nome} {self.__sobrenome}'


user1 = Usuario('Angelina', 'Jolie', 'angelina@gmail.com', '123456')
user2 = Usuario('Felicity', 'Jones', 'felicity@gmail.com', '654321')

print(user1.nome_completo())

print(Usuario.nome_completo(user1))

print(user2.nome_completo())

# Obs.: Para diferenciar os **métodos de classe** dos métodos de instância utilizamos um decorador (decorator)


class Usuario:
    contador = 0

    @classmethod
    def conta_usuarios(cls):  # não recebe o self como primeiro parâmetro e sim a própria classe (por convenção - cls)
        print(f'Classe: {cls}')
        print(f'Temos {cls.contador} usuário(s) no sistema')

    def __init__(self, nome, sobrenome, email, senha):
        self.__id = Usuario.contador + 1
        self.__nome = nome
        self.__sobrenome = sobrenome
        self.__email = email
        self.__senha = senha
        Usuario.contador = self.__id

    def nome_completo(self):
        return f'{self.__nome} {self.__sobrenome}'  # retornar com impressão


user = Usuario('Felicity', 'Jones', 'felicity@gmail.com', '123456')

Usuario.conta_usuarios()  # Forma correta
user.conta_usuarios()  # Possível, mas incorreta

"""
Obs.: Usamos métodos de instância quando esses métodos precisam acessar atributos. Quando não, usamos métodos de classe (menos usados).

Também podemos ter **métodos privados** (acessível apenas dentro da própria classe)
"""


class Usuario:
    contador = 0

    @classmethod
    def conta_usuarios(cls):  # não recebe o self como primeiro parâmetro e sim a própria classe (por convenção - cls)
        print(f'Classe: {cls}')
        print(f'Temos {cls.contador} usuário(s) no sistema')

    def __init__(self, nome, sobrenome, email, senha):
        self.__id = Usuario.contador + 1
        self.__nome = nome
        self.__sobrenome = sobrenome
        self.__email = email
        self.__senha = senha
        Usuario.contador = self.__id
        print(f'Usuário criado: {self.__gera_usuario()}')  # por aqui SIM, temos acesso ao método privado gera usuario

    def nome_completo(self):
        return f'{self.__nome} {self.__sobrenome}'

    # Usamos __ antes do nome do método para defini-lo como privado
    def __gera_usuario(self):
        return self.__email.split('@')[0]


# user = Usuario('Felicity', 'Jones', 'felicity@gmail.com', '123456')

print(user.__gera_usuario())  # por aqui, não temos acesso ao método __gera_usuario()... ele é privado

"""Vamos tornar este cenário mais interessante e próximo de uma aplicação real ... encriptando a senha"""


# !pip install passlib

# biblioteca para criptografar senhas

# cryp é um alias para pbkdf2_sha256

import time

class Usuario:
    contador = 0

    @classmethod
    def conta_usuarios(cls):  # não recebe o self como primeiro parâmetro e sim a própria classe (por convenção - cls)
        print(f'Classe: {cls}')
        print(f'Temos {cls.contador} usuário(s) no sistema')

    def __init__(self, nome, sobrenome, email, senha):
        self.__id = Usuario.contador + 1
        self.__nome = nome
        self.__sobrenome = sobrenome
        self.__email = email
        self.__senha = None  # cryp.hash(senha, rounds=200000, salt_size=16) # encripta senha: 200000 embaralhamentos, tamanho 16 chars
        Usuario.contador = self.__id
        print(f'Usuário criado: {self.__gera_usuario()}')

    def nome_completo(self):
        return f'{self.__nome} {self.__sobrenome}'

    def checa_senha(self, senha):
        # if cryp.verify(senha, self.__senha):  # decripta senha cadastrada e verifica se senhas (cadastrada e informada) são iguais
        #     return True
        return False

    def __gera_usuario(self):
        return self.__email.split('@')[0]

    def get_senha(self):
        return f'{self.__senha}'


nome = input('Informe o nome: ')
sobrenome = input('Informe o sobrenome: ')
email = input('Informe o e-mail: ')

while True:
    senha = input('Informe a senha: ')
    confirma_senha = input('Confirme a senha: ')

    if senha == confirma_senha:
        print('\n')
        print('Criando usuário...')
        time.sleep(2)
        user = Usuario(nome, sobrenome, email, senha)
        break
    else:
        print('Senhas não conferem, elas devem ser iguais... tente novamente!')

print('Usuário criado com sucesso!')

user.get_senha()

# Para efetuar login no sistema ...
password = input('Informe a senha para acesso: ')

if user.checa_senha(password):
    print('Acesso permitido')
else:
    print('Acesso negado')

# print(user.__senha())
print(user.get_senha())


# Exemplo da lâmpada
class Lampada:
    def __init__(self, voltagem, cor, luminosidade):
        self.__cor = cor
        self.__voltagem = voltagem
        self.__luminosidade = luminosidade
        self.__ligada = False

    def ligar_desligar(self):
        if self.__ligada:
            self.__ligada = False
        else:
            self.__ligada = True

    def checa_lampada(self):
        return self.__ligada


# Instância/Objeto
lamp1 = Lampada('branca', 110, 60)
lamp2 = Lampada('Azul', 220, 20)

lamp1.ligar_desligar()

print(f'A lâmpada está ligada? {lamp1.checa_lampada()}')

lamp2.ligar_desligar()

print(f'A outra lâmpada está ligada? {lamp2.checa_lampada()}')


class Cliente:

    def __init__(self, nome, cpf):
        self.__nome = nome
        self.__cpf = cpf

    def diz(self):
        print(f'O cliente {self.__nome} diz oi')


class ContaCorrente:
    contador = 4999

    def __init__(self, limite, saldo, cliente):
        self.__numero = ContaCorrente.contador + 1
        self.__limite = limite
        self.__saldo = saldo
        self.__cliente = cliente
        ContaCorrente.contador = self.__numero

    def mostra_cliente(self):
        print(f'O cliente é {self.__cliente._Cliente__nome}')


cli1 = Cliente('Angelina Jolie', '123.456.789-99')

cc = ContaCorrente(5000, 10000, cli1)

cc.mostra_cliente()

"""
O grande objetivo da POO é **encapsular** nosso código dentro de um grupo lógico e hierárquico utilizando classes.

* Encapsulamento: quando definimos elementos privados à classe que só são acessíveis por meio de métodos publicos.
* Abstração: ato de expor apenas dados relevantes de uma classe, escondendo atributos e métodos privados.

Palavras chave são "Controle de acesso".

Motivação: à medida que programas estruturados ficam maiores, mais complexas são a manutenção e a segurança do programa e a POO permite lida melhor com tal complexidade. O *encapsulamento* facilita a manutenção do código e aumenta a segurança com a proteção aos dados:

Outras vantagens POO:
- Mais flexivel: fácilidade ao descrever o mundo real através dos objetos;
- Aumento de produtividade: Redução das linhas de código programadas e maior Reutilização;
- Definição de responsabilidades com o conceito de classes;

Contrapartidas:
- Curva de aprendizagem mais prolongada.
- Difícil compreensão só com a teoria.

Último exemplo: **Sistema bancário**...

Primeiro, vamos construir o sistema baseado em uma única classe:
"""


class Conta:
    contador = 400

    def __init__(self, titular, saldo, limite):
        self.__numero = Conta.contador
        self.__titular = titular
        self.__saldo = saldo
        self.__limite = limite
        Conta.contador += 1

    def verifica_saldo(self):
        print(f'Saldo de {self.__saldo} do titular {self.__titular} com limite de {self.__limite}')

    def depositar(self, valor):
        if valor > 0:
            self.__saldo += valor
        else:
            print('O valor precisa ser positivo')

    def sacar(self, valor):
        if valor > 0:
            if self.__saldo >= valor:
                self.__saldo -= valor
            else:
                print('Saldo insuficiente')
        else:
            print('O valor deve ser positivo')

    def transferir(self, valor, conta_destino):
        # 1 - Remover o valor da conta de origem
        self.__saldo -= valor
        self.__saldo -= 10  # Taxa de transferência paga por quem realizou a transferência

        # 2 - Adicionar o valor na conta de destino
        conta_destino.__saldo += valor


conta1 = Conta('Josi', 150.00, 1500)

print(conta1.__dict__)

conta1.verifica_saldo()

conta1.depositar(150)

conta1.verifica_saldo()

conta1.sacar(200)

conta1.verifica_saldo()

conta2 = Conta('Vitor', 300, 2000)
conta2.verifica_saldo()

conta2.transferir(100, conta1)

conta1.verifica_saldo()
conta2.verifica_saldo()