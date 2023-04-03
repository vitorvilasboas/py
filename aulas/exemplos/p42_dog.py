"""
Crie um programa que contenha uma Função que verifica SE em uma frase informada pelo usuário contém a palavra 'dog' e outra
função que retorna o número de vezes que a palavra "dog" aparece na frase.
"""

def encontreCachorro(st):
    return 'dog' in st.lower().split()


def contaCachorro(st):
    count = 0
    for word in st.lower().split():
        if word == 'dog':
            count += 1
    return count


frase = input("Informe uma frase que mencione a palavra 'dog': ")
print(f" É {encontreCachorro(frase)} que a palavra 'dog' está na frase. ")
print(f" Nº de vezes em que a palavra 'dog' é mencionada: {contaCachorro(frase)}.")