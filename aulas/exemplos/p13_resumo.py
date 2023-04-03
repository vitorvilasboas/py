"""
Entrada/Saída padrão de dados

input() é a função de entrada de dados padrão (entrada via teclado)
print() é a função de saída de dados padrão (console)

Obs: Todos os comandos são encarados como funções a partir da versão 3.x do Python! 
"""

print('Avenida ' + 'Goiás')
print('Avenida ', 'Goiás')

print('Avenida ' + 15)     # ERRO (impossível somar texto com número)
print('Peso:' + 87.3)      # ERRO (impossível somar texto com número)

print('Avenida ', 15)
print('Peso:', 87.3)

print(10)
print('Idade:', 10)

nome = 'Vitor'
idade = 34
print(nome)
print(idade)

# print(nome + idade)  # ERRO
print(nome, idade)

print("Nome: ", nome, "| Idade: ", idade)

x = input("Informe seu nome: ")
print("nome: ", x)

"""
Exemplo: Crie um script que leia o nome de uma pessoa e mostre uma mensagem de boas-vindas como: 
“Seja bem vindo ao meu programa Fulano”.
"""

nome = input("Qual seu nome?")  # input() é a função de entrada de dados padrão (entrada via teclado)

print("Seja bem vindo ao meu programa ", nome)  # print() é a função de saída de dados padrão (console)

"""
Exemplo: Leia um nome e um sobrenome separadamente e depois concatene os dois apresentando o 
nome completo em um só comando.
"""

nome = input("Qual seu nome?")
sobrenome = input("Qual seu sobrenome?")
print("Bem vindo ", nome, sobrenome)

"""
Exemplo: Solicite o nome e a idade do usuário e então, envie a seguinte frase para o console: 
O seu nome é <nome> e a sua idade é <idade>.
"""

nome = input("Qual seu sobrenome?") # função de entrada padrão (do teclado)
idade = input("Qual sua idade?")
print("O seu nome é ", nome, " e a sua idade é ", idade, ".")

# ou

nome = input("Informe o nome: ") 
idade = int( input(nome + ", informe sua idade: ") ) # a função input só espera 1 parâmetro, por isso não usar a virgula(,)
print("O seu nome é ", nome, " e a sua idade é ", idade, ".")

nome = 'Vitor'
sobrenome = 'Vilas Boas'
idade = 10

print("Meu nome é ", nome, " e meu sobrenome é ", sobrenome, ". Tenho ", idade, " anos de idade.")   # função de saída padrão com concatenação por vírgula (,)

print("Meu nome é " + nome + " e meu sobrenome é " + sobrenome + ".")   # função de saída padrão com concatenação usando (+) - permitido somente para variáveis string

"""
CUIDADO: quando não há strings no argumento de saída o operador '+' define uma soma e 
não uma concatenação!
"""

print(7 + 4)    # soma entre números

print(7, 4)     # concatenação


# função de saída padrão (concatenação usando (+) para somar texto e números?)

print("Meu nome é " + nome + ". Tenho " + idade + " anos de idade.")    # ERRO

# SOLUÇÃO: usar vírgula (,) separando em argumentos ou converter os tipos numéricos para string

print("Meu nome é " +  nome + ". Tenho " + str(idade) + " anos de idade.")  # função de saída padrão com concatenação usando (+)

"""Comentários:
* servem para inserir observações no código, as quais não serão interpretadas/executadas. 
* SÃO IGNORADOS PELO INTERPRETADOR!
"""

# marcador de comentário em linha
# comentário em linha permite que você comente apenas uma linha/instrução

"""
Marcador de 
comentário 
em bloco (múltiplas linhas)
"""

"""
  Comentário em bloco
  permite que
  você comente
  várias
  linhas!!
"""

"""
Tipos de dados em Python:
Dado é um conjunto de valores "pura e simplesmente". 
Um dado pode ser um número, uma palavra, uma imagem, enfim, algo coletado, medido ou criado. 
Dados não precisam transmitir uma idéia! Por exemplo, o número 7, por si só, não transmite uma relação com nenhum contexto, é apenas o número 7. No entanto, ao definir o contexto, como dizer "Rua 7", o dado passa a transmitir uma ideia e passamos a ter uma **informação**.
Na computação, todo dado é representado por uma sequência de bits (ex: 0111).
O processamento e o armazenamento de um dado pelo computador depende de sua classificação (tipo do dado).

Tipos primitivos:
int define um valor inteiro...
float define um valor real (valores decimais)
bool define um valor boobleano (Verdadeiro ou Falso)
str define uma string (cadeia de caracteres - representa textos)

A TIPAGEM de variáveis no Python é DINÂMICA.
"""

a = 6       # atribuição de valor inteiro (int)

b = 5.97    # atribuição de valor real (float)
c = .4
d = 8.

e = "josé"  # atribuição de valor string (str) - aspas duplas
f = 'Maria' # atribuição de valor string (str) - aspas simples (boa prática)

g = False   # atribuição de valor booleano (bool)
h = True

print(type(a))

print(type(b))

print(type(c))

print(type(d))

print(type(e))

print(type(f))

print(type(g))

print(type(h))

"""
Exemplo: Solicite ao usuário informar um número e escreva a seguinte mensagem: 
"O número informado foi <número>."
"""

num = input("Informe um número: ")
print("O número informado foi ", num)

"""Python é **Case sensitive** (diferencia maiúsculas de minúsculas)"""

var = 'Texto'
Var = 'texTo'

"""
Conversão entre tipos de dados
"""

var = int(b)    # float to int

type(var)

var = float(a)  # int to float

var = str(b)    # int/float/bool to string

var = bool(0)   # int to bool (0=False, 1=True )

var = bool(1)

var = bool(2)   # int to bool (para qualquer valor diferente de 0, assume True)

var = bool()    # int to bool (quando nenhum argumento é informado, assume False)



# Operadores e expressões aritméticas

a = 8   # int
b = 3   # int
c = 7.  # float

soma = a + b

subtracao = c - b

produto = a * b   # multiplicação

divisao = a / b   # divisão comum

divisao_int = a // b  # divisão inteira (retorna apenas a parte inteira da divisão)

potencia = a**2    # potenciação (quadrado de a)

potencia = b**3

resto_div = a % 3  # resto da divisao de a por 3

print("Soma: ", soma)
print("Subtração: ", subtracao)
print("Produto: ", produto)
print("Divisão: ", divisao)
print("Divisão inteira: ", divisao_int)
print("Exponenciação: ", potencia)
print("Resto da divisão: ", resto_div)

"""
Exemplo: Leia dois números inteiros ou reais e mostre a soma, o produto e a média entre eles.
"""
num1 = float(input("Digite um número: "))
num2 = float(input("Digite outro número: "))
soma = num1 + num2
produto = num1 * num2
media = soma/2
print("O produto entre", num1, " e", num2, " é", produto, ", a soma é", soma, " e a média é", media, ".")

# Ordem de precedência padrão em expressões matemáticas (divisão/multiplicação --> soma/subtração --> módulo)

resultado = a + a - b * c

# Parênteses alteram a ordem de precedência padrão (primeiro o que está entre os parênteses mais internos)

resultado = ((a + a) - b) * c

"""
É possível somar textos (valores do tipo string)?
Resposta: Quase isso! É possível CONCATENAR (juntar) valores string
"""

name = 'Vitor '
surname = 'Vilas Boas'

fullName = name + surname   # nomes composto para variáveis seguem o padrão cammelcase

print(fullName)

var = '8' + '3'
print(var) # '83'

var = 8 + 3
var = int('8') + int('3')
print(var) # 11