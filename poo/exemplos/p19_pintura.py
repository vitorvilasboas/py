"""
Crie um script Python capaz de ler a largura e a altura de uma parede em metros, 
calcular e mostrar a área da parede, a quantidade de tinta necessária para pintá-la 
e o valor total gasto na pintura (em R$), sabendo que a tinta é vendida em latas de 
18 litros e que cada litro de tinta pinta uma área de 2 metros quadrados.
"""
largura = float(input("Informe a largura da parede em metros: "))
altura = float(input("Informe a altura da parede em metros: "))

area = largura * altura

qtd_litros = area / 2

qtd_latas = qtd_litros / 18

preco_lata = float(input("Informe o valor pago na lata de tinta de 18 litros: R$"))

gasto_total = preco_lata * qtd_latas

print("Valor total gasto na pintura da parede de", area,"m²: R$" + str(gasto_total))






