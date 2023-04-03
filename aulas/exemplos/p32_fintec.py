"""
Exemplo 010. Crie um script Python capaz de ler o nome do colaborador de uma FINTEC e a quantidade de clientes indicados por ele no último mês.
Sabendo que a vitor_poo paga a ele um salário fixo de R\$5000,00 por mês acrescido de um bônus de R\$150,00 a cada novo cliente indicado,
calcule e mostre o valor total do salário a que o colaborador tem direito a receber no mês (fixo + bônus).
"""

fixo = 5000
bonus = 150

nome = input("Informe o nome do colaborador: ")
qtd = int(input("Informe o número de clientes atendidos: "))

salario = fixo + (qtd * bonus)

print("O salário a receber do colaborador {} é: R${:.2f}".format(nome, salario))