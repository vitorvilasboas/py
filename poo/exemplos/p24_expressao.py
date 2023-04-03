"""
**Exemplo**: Crie um script em linguagem python que solicite 3 números inteiros e calcule o primeiro multiplicado pelo terceiro
dividido pelo segundo somado ao primeiro menos o terceiro.
"""

x = int(input("Digite o 1º número: "))
y = int(input("Digite o 2º número: "))
z = int(input("Digite o 3º número: "))

expr = x * z / (y + x) - z

print("Resultado: ", expr)