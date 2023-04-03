'''
Uma loja vende bicicletas com um acréscimo de 50% sobre o seu preço de custo. 
Ela paga a cada vendedor dois salários mínimos mensais, mais uma comissão de 15% 
sobre o preço de custo de cada bicicleta vendida, dividida igualmente entre eles. 
Crie um programa que leia o número de empregados da loja, o valor do salário mínimo, 
o preço de custo de cada bicicleta, o número de bicicletas vendidas, calcule e escreva: 
o salário final pago a cada empregado e o lucro líquido da loja.
'''

print("=== FORMOSA'S BICLETARIA === \n-- Módulo Financeiro:")
qtd_func = int(input("Informe o Nº de colaboradores na folha de pagamento: "))
salario_minimo = float(input("Informe o valor atual do salário mínimo: R$"))
custo_bicicleta = float(input("Informe o custo médio por bicicleta: R$"))
qtd_bicicleta = float(input("Informe a quantidade de bicicletas vendidas no mês:"))
preco_bicicleta = custo_bicicleta + (custo_bicicleta * 0.5)
# preco_bicicleta = custo_bicicleta * 1.05
comissao_total = (0.15 * custo_bicicleta * qtd_bicicleta)
comissao =  comissao_total / qtd_func
salario_final = (2 * salario_minimo) + comissao
lucro = ((preco_bicicleta - custo_bicicleta) * qtd_bicicleta) - (salario_final * qtd_func)

print("\n")
print("-- Custo unitário: R${:.2f}".format(custo_bicicleta))
print("-- Custo total: R${:.2f}".format(custo_bicicleta * qtd_bicicleta))
print("-- Venda total: R${:.2f}".format(preco_bicicleta * qtd_bicicleta))
print("-- Total comissão: R${:.2f}".format(comissao_total))
print("-- Comissão por colaborador: R${:.2f}".format(comissao))

print("-- Provimento por colaborador: R${:.2f}".format(salario_final))
print("-- Lucro da empresa: R${:.2f}".format(lucro))