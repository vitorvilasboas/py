"""
Você é parado por um policial por ser flagrado dirigindo rápido demais. Escreva uma função que, dada a velocidade de seu carro,
calcule o resultado da abordagem com base em 3 possibilidades: 'Advertência', 'Infração Leve" ou "Infração Grave".
* 'Advertência' -> caso a velocidade seja igual ou inferior a 60 Km/h.
* 'Infração Leve' -> caso a velocidade esteja entre 61 e 80 Km/h (inclusive).
* 'Infração Grave' -> caso a velocidade seja superior à 80 km/h.
Obs.: A função deve considerar um bônus de aniversário assim, caso seja seu aniversário, os limites de velocidade podem ser
      5km/h maiores em todos os casos.
"""


def consulta_infracao(velocidade, aniversario):
    if aniversario:
        velocidade = velocidade - 5
    else:
        velocidade = velocidade

    if velocidade > 80:
        return 'Infração Grave'
    elif velocidade >= 60:
        return 'Infração Leve'
    else:
        return 'Advertência'


# infracao = consulta_infracao(65, False)
# print(infracao)

infracao = consulta_infracao(81, True)
print(infracao)

# infracao = consulta_infracao(160, True)
# print(infracao)