"""
Exemplo 006. Dadas as variáveis
    Planeta = "Terra"
    Diametro = 12742
Use .format() para printar a seguinte frase:
    O diâmetro da terra é de 12742 kilômetros.
"""

planeta = "Terra"
diametro = 12742
print("O diâmetro da {} é {} kilômetros.".format(planeta,diametro))
print(f"O diâmetro da {planeta} é {diametro} kilômetros.")