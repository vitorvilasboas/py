'''
Crie um script em linguagem python que solicite 3 números inteiros e 
calcule o primeiro multiplicado pelo terceiro dividido pelo segundo 
somado ao primeiro menos o terceiro.
'''
n1 = int(input("Informe o 1º número inteiro: "))
n2 = int(input("Informe o 2º número inteiro: "))
n3 = int(input("Informe o 3º número inteiro: "))
x = n1 * (n3 / (n2 + (n1 - n3)))
print("Resultado: ", x)