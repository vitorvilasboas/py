"""
Exemplo 009. Crie um script Python capaz de mostrar o valor 3.1415926535 com 1, 2, 3 e 5 casas decimais.
"""

pi = 3.1415926535

print("Valor de PI com UMA casa decimal: %.1f" % pi)
print("Valor de PI com DUAS casas decimais: %.2f" % pi)
print("Valor de PI com TRÊS casas decimais: {:.3f}".format(pi))
print("Valor de PI com CINCO casas decimais: %.5f" % pi)


print(f"Valor de PI com CINCO casas decimais: {round(pi,5)}")
