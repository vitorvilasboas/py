"""
Desenvolva um programa que leia uma frase e informe quantas e quais vogais há nela.
"""

texto = input("Digite um texto qualquer: ").upper()
vogais = ['A','E','I','O','U']
encontradas = []
cont = 0
for v in vogais:
    n = texto.count(v)
    cont += n
    if n > 0:
        encontradas.append(v)

print(f"Vogais encontradas no texto: {encontradas}")
print(f"Quantidade de vogais encontradas: {cont}")