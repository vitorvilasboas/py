n1 = input("Informe o 1º número: ")
n2 = input("Informe o 2º número: ")

n1 = int(n1)
n2 = int(n2)

x = n1 ** (n2 + n1) * n1 / (n2 - n1)

## precedência padrão entre os operadores
# 1º exponenciação / resto
# 2º multiplicação/divisão
# 3º soma/subtração

print("Expressão: ", x)
