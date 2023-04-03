"""
Crie um script Python capaz de ler o valor que um funcionário ganha por hora 
e o número de horas trabalhadas no mês. Calcule e mostre o total do seu salário 
no referido mês, sabendo-se que são descontados 11% para o Imposto de Renda, 
8% para o INSS e 5% para o sindicato, faça um programa que nos dê: 
salário bruto; 
quanto pagou ao INSS; 
quanto pagou ao sindicato; 
o salário líquido (Salário Bruto - Descontos = Salário Líquido).
"""
print("\n=== CÁLCULO SALÁRIO FUNCIONÁRIOS ===", end='\n')
nome_func = input("\nInforme o nome do funcionário: ")
qtd_horas = int(input("\nInforme o nº de horas trabalhadas pelo funcionário " + nome_func + " no mes atual: "))
preco_hora = float(input("\nInforme o valor pago pela empresa por cada hora trabalhada: R$"))
salario_bruto = qtd_horas * preco_hora
desconto_irpf = salario_bruto * 0.11
desconto_inss = salario_bruto * 0.08
desconto_sind = salario_bruto * 0.05
total_descontos = desconto_irpf + desconto_inss + desconto_sind
salario_liquido = salario_bruto - total_descontos
print("\n## Contracheque de", nome_func + " ## \n\n",
	"Salário bruto: R$" + str(salario_bruto) + "\n\n", 
	"Descontos:\n", 
	"\t IRPF: R$" + str(desconto_irpf) + "\n", 
	"\t INSS: R$" + str(desconto_inss) + "\n", 
	"\t Sindicato: R$" + str(desconto_sind) + "\n", 
	"\t Total: R$" + str(total_descontos) + "\n\n", 
	"Salário liquido: R$" + str(salario_liquido))