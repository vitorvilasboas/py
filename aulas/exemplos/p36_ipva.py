'''
Crie um script Python capaz de ler o valor total de um veículo e o valor do seu IPVA, calcular e
mostrar a porcentagem do valor do IPVA sobre o valor total do veículo, usando o formato “O IPVA,
no valor de R$<valor_ipva>, corresponde a <porcentagem>% do valor total do veículo estimado
em R$<preco_veiculo>.”.
'''
preco_veiculo = float(input("Informe o preço do veículo: R$"))
valor_ipva = float(input(f"Informe o valor do IPVA do veículo que custa R${preco_veiculo}: "))
porcentagem = valor_ipva/preco_veiculo * 100
print("O IPVA, no valor de R${:.2f}, corresponde a {:.1f}% do valor total do veículo estimado em R${:.2f}".format(valor_ipva, porcentagem, preco_veiculo))