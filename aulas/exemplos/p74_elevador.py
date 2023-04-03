"""
Crie uma aplicação que simule o funcionamento de um elevador de um prédio. Uma classe denominada Elevador deve ser implementada para modelar os atributos (características) e ações (métodos) de um elevador.

* Dentre os atributos devem estar representados: o andar atual do elevador (térreo = 0), o número de andares no prédio (excluindo o térreo), a capacidade do elevador e a lotação atual (quantas pessoas estão presentes).
* Ao instânciar um objeto do tipo Elevador o construtor deve inicializar o andar atual como sendo equivalente ao térreo, lotação vazia e receber como parâmetros a capacidade do elevador e o total de andares do prédio.
* A aplicação deve possibilitar que o console que opera o elevador realize as seguintes ações:
   * entrar: para acrescentar uma pessoa ao elevador (somente se ainda houver espaço)
   * sair: para remover uma pessoa do elevador (somente se não estiver vazio
   * subir: para subir um andar (somente se não estiver no último andar)
   * descer: para descer um andar (somente se n

Os atributos no método construtor devem ser privados e acessíveis somente por meio de métodos públicos (encapsulamento).
"""

class Elevador:
	def __init__(self, andares, capacidade):
		self.andar = 0 # Térreo
		self.lotacao = 0
		self.andares = andares
		self.capacidade = capacidade
	def entrar(self):
		if self.lotacao == self.capacidade:
			print('Capacidade máxima atingida.')
			return
		self.lotacao += 1
	def sair(self):
		if not self.lotacao:
			print('Nenhuma pessoa dentro do elevador.')
			return
		self.lotacao -= 1
	def subir(self):
		if self.andar == self.andares:
			print('O elevador está no último andar.')
			return
		self.andar += 1
	def descer(self):
		if not self.andar:
			print('Elevador já está no último andar.')
			return
		self.andar -= 1
	def __repr__(self):
		chks = []
		d = self.__dict__
		for k in d:
			chks.append('%s=%s'%(k, d[k]))
		return '<Elevador |%s|>'%', '.join(chks)
elevador = Elevador(3, 10)
elevador.entrar()
print(elevador)
elevador.sair()
print(elevador)