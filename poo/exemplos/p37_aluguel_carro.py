'''
Crie um script Python capaz de ler a quantidade de Km percorridos por um carro alugado e a
quantidade de dias pelos quais ele foi alugado. Calcule o preço a pagar, sabendo que o valor do
aluguel é de R$50 ao dia mais R$3,50 por Km rodado.
'''
qtd_dias = int(input("Qual a quantidade de dias o veículo permaneceu alugado: "))
km_rodados = float(input(f"Quantos Km rodados nos {qtd_dias}: "))
preco = (qtd_dias * 50) + (km_rodados * 3.5)
print("Valor total do aluguel: R${:.2f}".format(preco))