"""
Leia uma quantidade indeterminada de números reais. A cada número lido o programa deve perguntar se o usuário deseja informar um novo número, ler e armazenar o próximo número enquanto a resposta for positiva. Ao final, assim que o usuário desejar encerrar a entrada de novos números, mostre:

A quantidade de números lidos;
A soma e a média dos números lidos;
A quantidade de números maiores ou iguais à média de todos os números lidos;
Quantos pertencem ao intervalo entre 10 e 50;
O maior e o menor número
Todos os números na ordem em que foram informados, um ao lado do outro
Todos os números na ordem inversa à que foram informados, um abaixo do outro
"""

def solicitar_numeros():
    # Função para solicitar os números e armazená-los numa lista
    lista = []  # lista vazia
    while (True):
        num = float(input("Informe um número real: "))
        lista.append(num)  # armazeno o número informado dentro da lista
        if input("Quer informar outro número? [S]im ou [N]ão: ").upper() == 'N':
            break
    return lista


def calcular_media(lista):
    # Função percorre a lista, soma todos os números e calcula a média entre eles
    soma = 0
    for n in lista:  # percorrer a lista
        soma += n
    media = soma / len(lista)
    return soma, media


def contar_maior_media(lista, media):
    # Conta a quantidade de números na lista que são maiores ou iguais à média
    cont = 0
    for n in lista:
        if cont >= media:
            cont += 1
    return cont


def contar_nums_intervalo(lista):
    # Conta a quantidade de números na lista que pertencem ao intervalo entre 10 e 50
    cont = 0
    for n in lista:
        if n >= 10 and n <= 50:
            cont += 1
    return cont


def extrair_menor(lista):
    # Retorna o menor número na lista
    menor = lista[0]
    for num in lista:
        if num < menor:
            menor = num
    return menor


def extrair_maior(lista):
    # Retorna o maior número na lista
    maior = lista[0]
    for num in lista:
        if num > maior:
            maior = num
    return maior


def imprime_lista(lista):
    # Mostra todos os números na lista na ordem em que foram informados, um ao lado do outro
    print("\nLista: ", end="")
    for i in range(0, len(lista), 1):
        print(lista[i], end=" ")


def imprime_lista_invertida(lista):
    # Mostra todos os números na lista na ordem inversa à que foram informados, um abaixo do outro
    print("\nLista invertida: ")
    for i in range(len(lista) - 1, -1, -1):
        print(lista[i])


# if __name__ == '__main__':

numeros = solicitar_numeros()
s, m = calcular_media(numeros)
qtd_maiorm = contar_maior_media(numeros, m)
qtd_intervalo = contar_nums_intervalo(numeros)
menor = extrair_menor(numeros)
maior = extrair_maior(numeros)

print(f"Foram informados {len(numeros)} números.")  # len(lista) é uma função nativa que retorna o tamanho da lista, ou seja, a qtd de elementos nela
print(f"A soma dos números informados é {s} e a média é {m}.")
print(f"Quantidade de números maiores ou iguais a média é {qtd_maiorm}.")
print(f"Quantidade de números entre 10 e 50 é {qtd_intervalo}.")
print(f"{menor} foi o menor número informado e {maior} o maior.")  # max(numeros) e min(numeros)

imprime_lista_invertida(numeros)
imprime_lista(numeros)