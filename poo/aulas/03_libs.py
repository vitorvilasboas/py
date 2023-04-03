""" Bibliotecas em Python (Módulos)
  * O python dispõe de uma lista de bibliotecas nativas, pré-programadas e prontas para uso, além de permitir a instalação de outras bibliotecas produzidas por outros programadores sem que seja necessário reprogramá-las no escopo do seu código.
  * Basta importar e usa (reaproveitamento de código)
  * Nativas: https://docs.python.org/3.7/library/index.html
  * Extras (criados pela comunidade) e disponibilizadas via PyPI: https://pypi.org/
"""

""" 
Importação de Bibliotecas Nativas:
"""
import math  # importa todas as funções da biblioteca
from math import *  # importa todas as funções da biblioteca
from math import sqrt, pow, ceil, floor, sin, cos, tan, tanh  # especifica funções a serem importadas
import math as m  # importação com apelido (alias)


# Biblioteca matemática math
import math

pot = math.pow(x, 3)  ## orientação a objetos
print(pot)

raiz = math.sqrt(4)
print(raiz)

log = math.log2(5)
print(log)

n = 1753.67
math.trunc(n)  # truncate (elimina parte flutuante)

xplus = math.ceil(1.56777)  # arredonda para cima

xmin = math.floor(1.55556)  # arredonda para baixo

# Exemplo: Leia os catetos de um triângulo retângulo e calcule sua hipotenusa.
# equação: h^2 = ca^2 + co^2  # hipotenusa ao quadrado é igual a soma dos quadrados dos catetos
import math
ca = input("Informe o valor do cateto adjacente: ")
co = input("Informe o valor do cateto oposto: ")
soma_cat = int(co) ** 2 + int(ca) ** 2  # soma dos quadrados dos catetos
h = math.sqrt(soma_cat)  # hipotenusa é igual à raiz quadrada da soma dos quadrados dos catetos
print(f"Sendo CA igual a {ca} e CO igual a {co} a hipotenusa é {int(h)}.")

# Exemplo: Leia um ângulo e calcule seu seno, cosseno e tangente.
from math import sin, cos, tan  # importando funções do módulo nativo math por demanda
angulo = float(input("Informe um ângulo: "))
seno = sin(angulo)
cosseno = cos(angulo)
tangente = tan(angulo)

# round(valor_a_aproximar, qtd_casas_decimais)
seno_ap = round(seno, 2)
cosseno_ap = round(cosseno, 2)
tangente_ap = round(tangente, 2)

print(f"O seno de {int(angulo)} é {seno_ap}.")
print(f"O cosseno de {int(angulo)} é {cosseno_ap}.")
print(f"A tangente de {int(angulo)} é {tangente_ap}.")

# Exemplo: Calcule o fatorial de um número fornecido pelo usuário
from math import factorial
x = input('Informe um número inteiro: ')
x = int(x)
fat = factorial(x)  # 5! = 5 * 4 * 3 * 2 * 1 = 120
print(f'O fatorial de {x} é {fat}.')

# Biblioteca time  (exclusiva para horários)

import time as t
start = t.time()
stop = t.time()
print(stop - start)

# Exemplo: Pergunte ao usuário seu nome completo e informe o tempo que o usuário levou para responder.
import time
inicio = time.time()
nome = input("Qual seu nome completo?")
final = time.time()
duracao = final - inicio
print(f"Ok {nome} você levou {round(duracao, 1)} segundos para responder.")

# Exemplo: Solicite um número do usuário e responda se ele está entre 0 e 1.
import time
real = float(input("Informe um número: "))
print("Processando, aguarde um instante...")
time.sleep(3)  # espera em segundos
if (real >= 0) and (real <= 1):  # as instruções dentro do bloco só são executadas se as duas condições foresm satisfeitas
    print(f"O número informado, {real}, está contido no intervalo entre 0 e 1.\n")
elif (real >= 1):
    print(f"O número informado, {real}, é positivo maior que 1.\n")
else:
    print(f"O número informado, {real}, é negativo.\n")
time.sleep(2)
print("Programa encerrado!")

# Biblioteca Datetime

import datetime

hoje = datetime.date.today()
print(hoje)

agora = datetime.datetime.now()
print(agora)

# Data formatada com a função strftime()
hoje = datetime.datetime.today().strftime('%d-%m-%Y %H:%M:%S')
print(hoje)

agora = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
print(agora)

# Convertendo Datetime para String com format
hoje = datetime.date.today()
data_em_texto = '{}/{}/{}'.format(hoje.day, hoje.month, hoje.year)
print(data_em_texto)

# Convertendo String em Datetime
data_e_hora = datetime.datetime.strptime(data_em_texto, '%d/%m/%Y')
print(data_e_hora)

# Forçando o fuso horário (timezone)
data_e_hora_atuais = datetime.datetime.now()
diferenca = datetime.timedelta(hours=-3)  # diferença de -3 horas, mais conhecida como UTC-3
fuso_horario = datetime.timezone(diferenca)
print(fuso_horario)

data_e_hora_sao_paulo = data_e_hora_atuais.astimezone(fuso_horario)
data_e_hora_sao_paulo_em_texto = data_e_hora_sao_paulo.strftime('%d/%m/%Y %H:%M')

print(data_e_hora_sao_paulo_em_texto)

# Biblioteca random

import random as rd

r = rd.random()  # atribui um número aleatório
print(r)

r = rd.randint(2, 8)  # atribui um número inteiro aleatório entre 2 e 8
print(r)

r = rd.uniform(2, 8)  # atribui um número real aleatório entre 2 e 8
print(r)

# Exemplo: Leia dois números inteiros e sorteie um número qualquer no intervalo entre os números fornecidos.
import random

num1 = int(input("Digite o primeiro número inteiro: "))
num2 = int(input("Digite o segundo número inteiro (maior que o primeiro): "))

sorteado = random.randint(num1, num2)

print("Numero sorteado: {}".format(sorteado))
# print(f"Numero sorteado: {sorteado}")
# print("Numero sorteado: ", sorteado)

# Exemplo: Faça o computador "pensar" em um número inteiro entre 1 e 10 e peça para o usuário tentar adivinhar qual foi o número escolhido pelo computador até acertar, ao final mostre quantos palpites foram necessários até adivinhar.
import random
comput = random.randint(1, 10)
usuario = int(input("Pensei em um número de 1 a 10, tente adivinhar qual foi: "))
tentativas = 1
while (usuario != comput):
    usuario = int(input("ERRADO, tente novamente: "))
    tentativas += 1
print(f"Isso, PARABÉNS, o número é {usuario}!")
print(f"Você precisou de {tentativas} chances para adivinhar.")

# Outras Bibliotecas

import os # (operational system)
print(os.path.dirname(__file__))
print(os.path.dirname(__file__) + '/teste.py')

# Instalação de uma biblioteca externa
# pip install mne  # instalação de módulos externos (não nativos)
# pip install pygame
import mne, pygame

# Exemplo PYGAME: Abra e reproduza o áudio de um arquivo MP3.
import pygame
pygame.init()
pygame.mixer.music.load('jimgle.mp3')
pygame.mixer.music.play()
pygame.event.wait()