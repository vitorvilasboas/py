"""
Crie um programa que imprima todos os números inteiros positivos inferiores a 1000 que são múltiplos 7.
"""
inicio = 1
fim = 1000

# para i no intervalo de 1 até 1001 passo 1 faça:
for i in range(inicio, fim+1, 1):
  if(i % 7 == 0):
    print(i, end=" ")

i = inicio
while(i <= fim):
  if(i % 7 == 0):
    print(i, end=" ")
  i += 1