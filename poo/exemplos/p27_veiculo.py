"""
Exemplo: Crie um script Python que leia os dados de um veículo (tais quais: marca, modelo, ano de fabricação,
capacidade de ocupantes e capacidade do tanque de combustível em litros) e mostre-os de forma organizada.
"""

print(" ####### Cadastro de veículos ####### ")
marca = input("Informe a marca: ")
modelo = input("Informe o modelo: ")
ano_fab = int(input("Informe o ano de fabricação: "))
capacidade_ocup = int(input("Informe a capacidade máxima de ocupação: "))
capacidade_tanque = float(input("Informe a capacidade do tanque de combustível (litros): "))

print("\nVeículo cadastrado com sucesso.\n\nDados do veículo:\nMarca =", marca, "\nModelo =", modelo, "\nAno de Fabricação =", ano_fab,
      "\nCapacidade Ocupantes =", capacidade_ocup, "\nCapacidade Tanque (lts) =", capacidade_tanque)