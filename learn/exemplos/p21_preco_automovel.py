"""
Crie um script Python capaz de ler o modelo de um automóvel e seu preço de fábrica. 
Sabendo que o preço final do automóvel é definido pela soma entre o preço de fábrica, 
o valor dos impostos (45% do preço de fábrica) e a percentagem do revendedor 
(28% do preço de fábrica), calcule e mostre o preço final do automóvel.
"""
modelo = input("Informe o modelo do automóvel: ")
preco_fabrica = float(input("Informe o preço de fabrica do automóvel modelo " + modelo + ": R$"))
impostos = preco_fabrica * 0.45
perc_rev = preco_fabrica * 0.28
preco_final = preco_fabrica + impostos + perc_rev
print("Preço final do automóvel modelo", str(modelo) + ": R$" + str(preco_final))