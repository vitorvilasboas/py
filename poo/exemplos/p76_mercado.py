"""
Desenvolver uma aplicação de uma loja virtual que, ao ser inicializada, apresente ao usuário um menu para que ele escolha entre cadastrar novos produtos, listar produtos cadastrados, comprar produtos, visualizar carrinho de compras ou sair da aplicação. Ao adicionar um produto no carrinho de compras, deve-se verificar se já existe um mesmo produto no carrinho e alterar a
quantidade desse item no carrinho caso ele já exista. Ao finalizar a compra deve ser apresentado ao usuário o total a pagar pela compra de acordo com os produtos e quantidades constantes no carrinho de compra.
"""

from os import system

class Product:
	def __init__(self, id, name, price, qunt):
		self.id = id
		self.name = name
		self.price = price
		self.qunt = qunt

class Store:

	def __init__(self):
		self.products = []
		self.cart = []

	@staticmethod
	def table(data, i = 2):
		labels = [_['label'] for _ in data]
		lines = [_['lines'] for _ in data]
		lx = [[line[x] for line in lines] for x in range(len(lines[0]))]
		doLine = lambda: '|%s|'%'|'.join(['{:<%s}'%(len(x) + i) for x in labels])
		doSep = lambda: '-' * s
		s = 0
		for l in labels:
			s += len(l) + i + 1
		s += 1
		print(doSep())
		print(doLine().format(*labels))
		print(doSep())
		for x in lx:
			print(doLine().format(*x))
		print(doSep())

	def fetchAllProp(self, prop, haystack):
		return [x.__dict__[prop] for x in haystack]

	def listProducts(self):
		if not len(self.products):
			print('Não há produtos cadastrados.')
			return
		Store.table([
			{
				'label': 'ID',
				'lines': self.fetchAllProp('id', self.products)
			},
			{
				'label': 'NOME',
				'lines': self.fetchAllProp('name', self.products)
			},
			{
				'label': 'PREÇO',
				'lines': self.fetchAllProp('price', self.products)
			},
			{
				'label': 'QUANTIDADE',
				'lines': self.fetchAllProp('qunt', self.products)
			}
		], 10)

	def listCart(self):
		if not len(self.cart):
			print('Não há produtos no carrinho.')
			return
		Store.table([
			{
				'label': 'ID',
				'lines': self.fetchAllProp('id', self.cart)
			},
			{
				'label': 'NOME',
				'lines': self.fetchAllProp('name', self.cart)
			},
			{
				'label': 'PREÇO',
				'lines': self.fetchAllProp('price', self.cart)
			},
			{
				'label': 'QUANTIDADE',
				'lines': self.fetchAllProp('qunt', self.cart)
			}
		], 10)
		print('Total a pagar: R$ %d'%self.totalToSell())

	def addProduc(self):
		name = getInput('Digite o nome do novo produto >> ')
		has = Store.getProduct(name.lower(), self.products)
		if has:
			print('Esse produto já existe. Incrementando quantidade.')
			has.qunt += 1
			return
		price = toInt('Agora, o preço >> ', True)
		qunt = toInt('E a quantidade >> ')
		self.products.append(Product(len(self.products), name, price, qunt))
		print('Produto adicionado.')

	@staticmethod
	def getProduct(what, haystack):
		for p in haystack:
			n = p.name.lower()
			if what == n or what == p.id:
				return p
		return None

	def buyProduct(self):
		i = Store.getProduct(toInt('Digite o id do produto desejado >> ', a = True), self.products)
		if i is None:
			print('Produto não encontrado.')
			return
		if not i.qunt:
			print('Desculpe, sem estoque suficiênte.')
			return
		i.qunt -= 1
		print('Produto comprado.')

	def addToCart(self):
		i = Store.getProduct(toInt('Digite o id do produto desejado >> ', a = True), self.products)

		if i is None:
			print('Produto não encontrado.')
			return

		if not i.qunt:
				print('Estoque insuficiênte para adicionar ao carrinho.')
				return

		hasInCart = Store.getProduct(i.id, self.cart)

		if hasInCart:
			print('Produto já existe no carrinho, adicionando 1 unidade.')
			hasInCart.qunt += 1
			i.qunt -= 1
		else:
			self.cart.append(Product(i.id, i.name, i.price, 1))
			i.qunt -= 1
			print('Produto adicionado ao carrinho.')

	def delInCart(self):
		while len(self.cart):
			del self.cart[0]

	def totalToSell(self):
		return sum([c.price * c.qunt for c in self.cart])

	def buyInCart(self):
		if not len(self.cart):
			print('Adicione algo ao carrinho primeiro!')
			return
		print('Esses são os produtos a serem comprados: ' + ', '.join([x.name for x in self.cart]))
		print('Total: %d'%self.totalToSell())
		if not confirm('Dejesa mesmo finalizar a compra? [S/N] >> '):
			print('Operação cancelada.')
			return
		self.delInCart()
		print('Compra finalizada.')

def getInput(t):
	r = None
	while not r:
		r = input(t)
	return r
def toInt(t, f = False, a = False):
	cls = int if not f else float
	try:
		t = cls(input(t))
		if not a and t <= 0:
			return 1
		return t
	except ValueError:
		print('Número inválido')
	return toInt(t, f, a)

s = Store()

def help():
	print('Uso:')
	for k in actions:
		v = actions[k]
		print('[%s] %s'%(k, v[1]))

def clear():
	system('cls || clear')

def exit_():
	print('\n=== PyMarket - Saindo ===')
	exit(0)

confirm = lambda t: getInput(t).lower() in 'sy'

actions = {
	'1': (s.listProducts, 'Listar produtos'),
	'2': (s.addProduc, 'Adicionar produto'),
	'3': (s.buyProduct, 'Comprar'),
	'4': (s.addToCart, 'Adicionar ao carrinho'),
	'5': (s.listCart, 'Listar carrinho'),
	'6': (s.buyInCart, 'Finalizar compra do carrinho.'),
	'7': (clear, 'Limpar console'),
	'0': (exit_, 'Sair do sistema'),
	'h': (help, 'Ajuda')
}

print('=== PyMarket ===')

help()

while True:
	try:
		c = input('> ')
		if not c:
			continue
		elif c not in actions:
			print('Comando inválido. Use h para obter ajuda.')
		else:
			actions[c][0]()
	except KeyboardInterrupt:
		exit_()