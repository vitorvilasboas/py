"""
Com base no exemplo desenvolvido e apresentado na aula sobre POO, desenvolva uma nova versão do sistema bancário que, ao ser inicializada, apresente um menu ao usuário para que ele escolha o que deseja realizar: criar uma conta, efetuar saque, efetuar depósito, efetuar transferência, listar contas ou sair do sistema.
"""

class Conta:

  contador = 400

  def __init__(self, titular, saldo, limite):
    self.__numero = Conta.contador
    self.__titular = titular
    self.__saldo = saldo
    self.__limite = limite
    Conta.contador += 1

  def getTitular(self):
    return self.__titular

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

from os import system

class Banco:

    # Simula um login, para realizar depósitos, transferências e saques
    def login(self):
        if self.contaLogada is not None:
            print('\n[+] Já está logado como %s'%self.contaLogada.titular)
            if not confirm('[+] Deseja logar em outra? [S/N] > '):
                return
        if not len(self.users):
            print('\n[x] Nenhum usuário existente.')
            return
        self.listUsers()
        c = self.getUser(getInput('Digite a conta a qual quer logar >> '))
        if c is None:
            print('\n[x] Usuário inválido')
            return
        self.currentUser = c
        print('\n[+] Logado como %s'%self.currentUser.name)

    # # Retorna um objeto do tipo user da lista de usuários
    # def getConta(self, name):
    #     for user in self.users:
    #         if user.name == name:
    #             return user
    #     return None

    # # Verifica se usuário está logado, e retorna True se sim caso não, emite uma mensagem e retorna False
    # def isLogged(self, a):
    #     if self.currentUser:
    #         return True
    #     print('\n[x] Impossível %s: logue em alguma conta.'%a)
    #     return False

    # Método que realiza o depósito.
    def deposit(self):
        if not self.isLogged('depositar'):
            return
        q = toInt('Digite a quantidade >> ')
        self.currentUser.saldo += q
        print('\n[+] Depositado com sucesso.')

    # Método que realiza saque.
    def saq(self):
        if not self.isLogged('sacar'):
            return
        q = toInt('Digite a quantidade a sacar >> ')
        self.currentUser.saldo -= q
        print('\n[+] Saque realizado.')

    # Lista os usuários.
    def listUsers(self):
        if not len(self.users):
            print('\n[x] Nenhum usuário existente.')
            return
        print('\n=== Usuários ===')
        for user in self.users:
            print('> %s - R$ %d'%(user.name, user.saldo))
        print()

    # Método que transfere entre contas.
    def transf(self):
        if not self.isLogged('transferir'):
            return
        para = getInput('Digite a conta de destino >> ')
        if not self.hasUser(para):
            print('\n[x] Usuário inexistente.')
            return
        if para == self.currentUser.name:
            print('\n[x] Não pode transferir para si mesmo.')
            return
        para = self.getUser(para)
        q = toInt('Digite a quantidade a transferir >> ')
        self.currentUser.saldo -= q
        para.saldo += q
        print('\n[+] Transferência concluída.')

contas = []
contaLogada = None

def conta_existe(cpf):
  for conta in contas:
    if conta.cpf == cpf:
      return True
  return False

def add_conta(cpf):
  if not conta_existe(cpf):
    saldo = input('Informe o saldo inicial: R$')
    limite = input('Informe o limite inicial: R$')
    contas.append(Conta(cpf, saldo, limite))
    print('\n[+] Conta criada com sucesso.')
  else:
    print('\n[x] Já existe uma conta vinculada ao cpf.')


if __name__ == '__main__':
  # Executa comandos até usuário sair
  while True:
    print('=== Meu Banco ===')

    if contaLogada:
      print(f'''
      -------------------------
      | [1] Adicionar conta   |
      | [2] Listar contas     |
      | [3] Depósito          |
      | [4] Saque             |
      | [5] Transferência     |
      | [0] Sair              |
      -------------------------
      ''')

      opcao = None
      while not opcao:
        opcao = input('Selecione uma opção >> ')
      system('cls || clear') # Limpa o console
      if opcao == 0:
        print('\n[+] Saindo...!')
        exit()
      elif opcao == 1:
        cpf = input('Digite o CPF do titular:')
        add_conta(cpf)
      elif opcao == 2:
        pass
      elif opcao == 3:
        pass
      elif opcao == 4:
        pass
      elif opcao == 5:
        pass
      else:
        print('\n[x] Opção inválida.')
        # return

    # actions = {'0': banco.exit, '1': banco.addUser, '2': banco.listUsers, '3': banco.login, '4': banco.deposit, '5': banco.saq, '6': banco.transf }
    # actions[option]()