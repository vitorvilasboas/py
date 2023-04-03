"""
Exemplo 002: Leia uma temperatura em graus Celsius e apresente-a convertida em graus Fahrenheit.
Sabendo que F= (9*C+160)/5, sendo F a temperatura em Fahrenheit e C em Celsius.
"""

C = input("Digite uma temperatura em Celsius: ") # string
F = (9 * float(C) + 160)/5
print(C, "º Celsius equivale a ", F, "º Fahrenheit.")