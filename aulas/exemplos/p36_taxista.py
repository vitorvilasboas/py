'''
Um motorista de táxi deseja calcular o rendimento de seu carro na praça. Sabendo-se que 
o preço do combustível é de R$7.00, escreva um programa para: ler a marcação do odômetro 
no início e no final do dia em Km, o número de litros de combustível gasto e o valor total (R$) 
recebido dos passageiros; calcular e escrever a média do consumo em Km/L e o lucro líquido do dia.
'''
print("\n=== Cálculo Rendimento TÁXI ===")
km0 = float(input("Odômetro inicial (Km): "))
kmf = float(input("Odômetro final (Km): "))
qtd_litros = float(input("Combustível injetado (litros): "))
ganho = float(input("Total recebido no dia: R$"))

km_rodados = kmf - km0
consumo = km_rodados / qtd_litros

gasto = qtd_litros * 7.00
lucro = ganho - gasto

print("\nGasto do dia: R${:.2f}".format(gasto))

print("Lucro do dia: R${:.2f}".format(lucro))
print("Consumo médio veículo: {:.1f} km/l".format(consumo))