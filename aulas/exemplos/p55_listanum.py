"""
Crie um programa que pergunte ao usuário quantos números inteiros ele deseja informar e
leia cada um desses números. Ao final, apresentar a soma de todos os números lidos.
"""
num = int(input("Informe a quantidade de inteiros a ser informada: "))
# print("Quantidade informada: ", num)
print(f"Quantidade informada: {num}")

soma = 0
lista = []
for i in range(0, num):
  n = int(input(f"Informe o {i+1}o número: "))
  lista.append(n)
  soma = soma + n
print(f"Lista de números fornecidos: {lista}")
print(f"A soma de todos os {num} numeros informados é {soma}.")