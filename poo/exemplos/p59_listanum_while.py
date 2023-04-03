"""
Crie um programa que leia um conjunto de números inteiros e armazene-os em uma lista.
Admita o valor 999 como condição de parada para fim de leitura dos números.
Ao final, imprima a quantidade de números informados e a média entre eles. .
"""
lista = []
while True:
  num = int(input("Informe um número inteiro (999 para sair): "))
  if num != 999:
    lista.append(num)
  else:
    break
print(f'Números informados: {lista}')
media = (sum(lista)/len(lista))
print(f"Foram informados {len(lista)} números, sendo {media} a média entre eles.")

## Perguntando se o usuário deseja continuar (Sim/Não
lista = list()
while True:
    lista.append(int(input('Informe um número: ')))
    r = str(input('Continuar? (y=sim, n=não)'))
    if r == 'ý':
        break
print(f'Números informados: {lista}')
media = (sum(lista)/len(lista))
print(f"Foram informados {len(lista)} números, sendo {media} a média entre eles.")