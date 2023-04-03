"""
Crie uma função chamada area( ) que receba as dimensões de um terreno retangular (largura e comprimento),
calcule e retorne a area do terreno.
"""


def area_terreno(l, c):
    area = l * c
    return area


largura = float(input("Informe a largura do terreno: "))
comprimento = float(input("Informe o comprimento do terreno: "))

resultado = area_terreno(largura, comprimento)  # chamada da função

print(f"Área terreno = {resultado}")