"""
Crie um programa que simule o controle do volume e a troca de canais de uma Televisão a partir de um Controle Remoto. O sistema deve permitir que o usuário aumente ou diminua o volume da Televisão por meio do controle remoto uma unidade de volume por vez. Da mesma forma deve ocorrer com o número do canal, que pode aumentar ou diminuir um número por vez. A cada vez que o volume ou o canal forem alterados, o sistema deve informar ao usuário o valor atual desses atributos. O uso de encapsulamento é obrigatório nos atributos do método construtor.
"""

class Tv:
	def __init__(self):
		self.__channel = 0 # __ no ínicio de algum atributo o deixa privado a classe.
		self.__volumn = 0
	def proximoCanal(self):
		self.__channel += 1
		print(self.__repr__())
	def voltarCanal(self):
		self.__channel -= 1
		print(self.__repr__())
	def aumentarVolume(self):
		self.__volumn += 1
		print(self.__repr__())
	def baixarVolume(self):
		self.__volumn -= 1
		print(self.__repr__())
	def __repr__(self):
		return f'Tv [Volume={self.__volumn}|Canal={self.__channel}]'
tv = Tv()

actions = {
	'a': tv.aumentarVolume,
	's': tv.baixarVolume,
	'd': tv.proximoCanal,
	'f': tv.voltarCanal
}

def help():
	print(
		'\nComo usar:\n'
		'a -> aumentar volume\n'
		's -> baixar volume\n'
		'd -> Aumentar 1 canal\n'
		'f -> Baixar um canal\n'
		'q -> Desligar Tv\n'
		'h -> Ajuda\n'
	)

print('=== Controle ===')

help()

while True:
	k = input('control> ')
	if not k:
		continue
	elif k in actions:
		actions[k]()
	elif k == 'q':
		print('Desligando Tv...')
		break
	elif k == 'h':
		help()
	else:
		print('Comando inválido, use h para ajuda.')

