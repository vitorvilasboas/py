'''
Crie um programa em python que calcule e mostre o índice de massa 
corpórea (IMC) do usuário, sabendo que IMC = peso/(altura*altura).
'''
nome = input("Informe seu nome: ")
peso = float(input(nome + ", informe seu peso: "))
altura = float(input(nome + ", informe sua altura: "))

imc = peso / (altura**2)

print(nome, "seu IMC foi calculado em", str(imc) + ".")


