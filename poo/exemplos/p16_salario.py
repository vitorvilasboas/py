'''
Uma empresa de softwares paga a seus desenvolvedores um salário fixo de 
R$3000,00 por mês, mais um bônus de R$300,00 a cada novo cliente incorporado 
pela empresa naquele mês. Desenvolva um algoritmo que leia o nome do 
desenvolvedor, o mês corrente, a quantidade de clientes incorporados no mês e 
calcule o salário total do desenvolvedor (fixo + bônus). 
Ao final, elabore uma frase única contendo os valores do nome do 
desenvolvedor e de seu salário total no referido mês.
'''
FIXO, BONUS = 3000.00, 300.00
nome = input("Informe o nome do colaborador: ")
mes = input("Informe o mês atual por extenso: ")
clientes = int(input("Informe o número de clientes incoporados pelo colaborador " + nome + " durante o mês de " + mes + ": "))
salario = FIXO + (BONUS * clientes)
print("O valor total do salário colaborador", nome, ", incluindo bônus, é: R$" + str(salario))
# print("O valor total do salário colaborador", nome, ", incluindo bônus, é: R${:.2f}".format(salario))