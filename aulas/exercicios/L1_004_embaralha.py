"""
Desenvolva um programa com uma função que receba uma string como parâmetro e devolva outra string com os caracteres embaralhados. Por exemplo: se a função receber a palavra python, pode retornar npthyo, ophtyn ou qualquer outra combinação possível, de forma aleatória. Padronize a entrada e a saída da função com caracteres em caixa alta.
"""

def embaralhar(texto):
    import random
    t = list(texto)
    random.shuffle(t)
    return ''.join(t)

print(embaralhar(input("Digite um texto qualquer: ").upper()))