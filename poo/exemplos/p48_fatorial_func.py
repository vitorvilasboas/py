"""
Exemplo: Sabendo que o fatorial de um número é igual ao produto dele com seus antecessores inteiros positivos (e.g. 5! = 5 x 4 x 3 x 2 x 1 = 120), desenvolva um programa com uma função em que, dado um inteiro não-negativo num, retorne o fatorial de num (num!).
"""

fat = 1

def fatorial(num: int):
    global fat
    for i in range(num, 0, -1):
        fat *= i
    return fat


f = int(input("Informe o número na qual deseja saber o fatorial: "))
# valor = f
if f >= 0:
    fatorial(f)
    print(f"O fatorial de {f} é {fat}")
else:
    print("\033[0;31;31m|ERROR| o número informado é negativo")