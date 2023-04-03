"""
Crie um script usando funções para encontrar as raizes de uma **equação do segundo grau (ax² + bx + c = 0)
a partir da fórmula de báscara. São três soluções possíveis conforme o valor do discriminante (delta):
> delta > 0, a equação possui duas raízes reais e distintas
> delta = 0, a equação possui raízes reais iguais
> delta < 0, a equação não possui raízes reais
"""
# delta = b**2 - 4*a*c
def calculo_delta(a, b, c):
    delta = b ** 2 - 4 * a * c
    return delta


# X = (-b +- raiz(delta))/(2*a)
def calculo_raizes(a, b, delta):
    import math
    if delta >= 0:
        x1 = (-b + math.sqrt(delta)) / 2 * a  # math domain error
        x2 = (-b - math.sqrt(delta)) / 2 * a
    else:
        x1, x2 = None, None
    return x1, x2


a = float(input("Informe o valor do coeficiente a: "))
b = float(input("Informe o valor do coeficiente b: "))
c = float(input("Informe o valor do coeficiente c: "))

delta = calculo_delta(a, b, c)
raiz1, raiz2 = calculo_raizes(a, b, delta)

if raiz1 is None:
    print("Não existem raizes reais para os coeficiente informados.")
else:
    print(f"Raiz 1 = {raiz1} | Raiz 2 = {raiz2}")