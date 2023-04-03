# ===============
# STRINGS
# ===============
"""
texto = "Aula sobre strings em linguagem Python"
texto[0]    # primeiro elemento
texto[2]    # terceiro elemento
texto[-1]   # último elemento
texto[-2]   # penúltimo elemento
texto[5:10]     # intervalo índice 5 até 9
texto[:8]   # até índice 7
texto[8:]   # a partir do índice 8
texto[3:30:2]   # intervalo inicio, final, passo
texto[3::2]     # intervalo inicio, passo
len(texto)  # tamanho
texto.lower()   # minusculo
texto.upper()   # maiusculo
texto.capitalize()  # primeira maiuscula
texto.title()   # primeiraS maiusculaS
texto.count('o')    # conta o
texto.count('o', 2, 10) # conta o no intervalo
texto.find('sobre')     # busca ( Sim=indice | Não=-1)
texto.index('l')    # índice de 'l'
texto.replace('Python','Java')  # substitui
'Python' in texto   # True/False
'Python' not in texto   # True/False
texto.split()   # fatia nos espaços e separa termos numa lista
lista = texto.split('e')    # fatia nos 'e' e separa termos numa lista
'e'.join(lista)     # junta novamente
texto.strip()   # elimina todos os espaços desnessários
texto.rstrip()  # (right strip) elimina espaços desnecessários à direita
texto.lstrip()  # (left strip) elimina espaços desnecessários à esquerda
texto.isalpha()     # só letras?
texto.isnumeric()   # só números?
texto.isalnum()     # só letras e números?
"""

"""
Exemplo: Leia um número e informe se o dado informado é realmente um número.
"""
print(f"O dado informado é numérico?... {input('Informe um número qualquer: ').isnumeric()}.")

"""
Exemplo: Solicite o nome do usuário e dê boas vindas somente se alguma coisa for digitada, informe caso não tenha sido digitado nada.
"""
nome = input("Qual seu nome: ")
print(f"Nada foi informado!" if nome is '' else f"Sério!?... Bem vindo, {nome}!")

"""
Exemplo: Construa uma função que receba um e-mail e retorne apenas o domínio do e-mail. Por exemplo, passando como parâmetro "loriosmenilson@gmail.com" retornaria: gmail.com
"""
# def obterDominio(email):
#     return email.split('@')[-1]
# obterDominio(input("Informe seu e-mail: "))

"""
Exemplo: Leia o nome completo de uma pessoa e imprima:
* o nome com todas as letras maiúsculas
* as partes do nome separadas (primeiro nome, segundo nome, terceiro nome...)
* quantas letras 'o' e 's' têm no nome completo
* quantos elementos têm o primeiro nome
"""
# nome = input('Escreva seu nome completo: ')
# mai = nome.upper()
# sep = nome.split('.')
# os = nome.count('o')
# ss = nome.count('s')
# elements = len(sep[0])
# print(f"Seu nome completo em maiúsculo: {mai}")
# print(f"As partes do seu nome separadas são, respectivamente: {sep}")
# print(f"Ocorrências da letra o: {os}")
# print(f"Ocorrências da letra s: {ss}")
# print(f"Número de elementos no primeiro nome: {elements}")

"""
Exemplo: Solicite uma senha do usuário e mostre:
* se a senha começa com a palavra "eu"
* se a senha possui a palavra "senha"
* se a senha possui letras, números e caracteres especiais
"""
# senha = input("Digite uma senha: ")
# eu = 'eu' in senha[0:2]
# sen = 'senha' in senha
# letra = senha.isalpha()
# numero = senha.isnumeric()
# lnc = senha.isalnum()
# print(f"A senha começa com a palavra 'eu': {eu}")
# print(f"A senha possui a palavra 'senha': {sen}")
# print(f"A senha possui apenas letras: {letra}")
# print(f"A senha possui apenas números: {numero}")
# print(f"A senha não possui caracteres especiais: {lnc}")

"""
Exemplo: Peça para o usuário informar uma frase qualquer e depois uma palavra qualquer e mostre:
* o comprimento da palavra informada
* o caractere na frase cujo índice é igual ao comprimento da palavra
* em que posição a letra "e" aparece na frase pela primeira vez
* a frase com sua primeira palavra substituída pela palavra fornecida pelo usuário
"""
# frase = input("Escreva uma frase qualquer: ")
# palavra = input("Escreva uma palavra qualquer: ")
# compp = len(palavra)
# relac = frase[compp]
# e = frase.find("e")
# split = frase.split()
# subtt = frase.replace(split[0], palavra)
# print(f"O comprimento da palavra escrita é: {compp}")
# print(f"O caractere que corresponde ao numero do comprimento da palavra é: {relac}")
# print(f"A letra 'e' aparece pela primeira vez na frase no índice {e}")
# print(subtt)
