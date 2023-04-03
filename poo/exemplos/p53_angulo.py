"""**Exemplo**. Leia um ângulo e calcule seu seno, cosseno e tangente."""

from math import sin, cos, tan # importando funções do módulo nativo math por demanda

angulo = float(input("Informe um ângulo: "))

seno = sin(angulo)
cosseno = cos(angulo)
tangente = tan(angulo)

# round(valor_a_aproximar, qtd_casas_decimais)
seno_ap = round(seno, 2)
cosseno_ap = round(cosseno, 2)
tangente_ap = round(tangente, 2)

print(f"O seno de {int(angulo)} é {seno_ap}.")
print(f"O cosseno de {int(angulo)} é {cosseno_ap}.")
print(f"A tangente de {int(angulo)} é {tangente_ap}.")