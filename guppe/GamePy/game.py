"""
Desenvolver uma aplicação que simule um jogo em que, ao ser inicializada, seja solicitado ao usuário o nível de dificuldade e, sem seguida, gere, de forma aleatória, um cálculo matemático qualquer usando operações de soma, subtração ou multiplicação. O objetivo do jogo é que o usuário acerte o resultado do cálculo apresentado. Caso o usuário acerte o resultado, o sistema deve incrementar 1 a sua pontuação (score). Acertando ou errando o resultado, ele poderá ou não continuar o jogo.
"""

from models.calculo import Calculo


def main() -> None:
    pontos = 0
    jogar(pontos)
    # return None


def jogar(pontos: int) -> None:
    dificuldade: int = int(input('Informe o nível de dificulade desejado [1, 2, 3 ou 4]: '))

    calc = Calculo(dificuldade) # chamma o método construtor

    print('Informe o resultado para a seguinte operacação: ')
    calc.mostrar_operacao()

    resposta: int = int(input())

    if calc.checar_resultado(resposta):
        pontos += 1
        print(f'Você tem {pontos} pontos(s).')

    continuar: int = int(input('Deseja continuar no jogo? [1 - sim, 0 - não] '))

    if continuar:
        jogar(pontos)
    else:
        print(f'Você finalizou com {pontos} ponto(s).')
        print('Até a próxima!')


if __name__ == '__main__':
    main()

