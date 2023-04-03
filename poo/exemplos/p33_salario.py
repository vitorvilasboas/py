"""
Exemplo 011. Leia um salario (real) e um percentual (inteiro) de aumento. Informe o salario com aumento em duas casas decimais.
"""

salario = float(input("Informe o salario (R$): "))
aumento = float(input("Informe o valor do aumeto (%): "))

calculo = salario * aumento/100
salario += calculo       # salario = salario + calculo

print(f"Novo salário: R${round(salario,2)}")