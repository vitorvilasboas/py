"""
Desenvolver uma aplicação que simule um jogo em que, ao ser inicializada, seja solicitado ao usuário o nível de dificuldade e, sem seguida, gere, de forma aleatória, um cálculo matemático qualquer usando operações de soma, subtração ou multiplicação. O objetivo do jogo é que o usuário acerte o resultado do cálculo apresentado. Caso o usuário acerte o resultado, o sistema deve incrementar 1 a sua pontuação (score). Acertando ou errando o resultado, ele poderá ou não continuar o jogo."""

from random import randint, choice
from os import system


class Game:
    difficult = {
        0: (1, 10),  # Easy
        1: (50, 100),  # Medium
        2: (100, 200)  # Hard
    }
    eqs = ['+', '-', '*']
    congr = {
        5: 'Está indo bem!',
        12: 'Continue assim!',
        14: 'Cê é o bixão memo hein doido :o'
    }

    def __init__(self):
        self.score = 0
        self.level = 0
        self.question = 0

    def genEq(self):
        equ = '%d %s %d' % (
        randint(*self.difficult[self.level]), choice(self.eqs), randint(*self.difficult[self.level]))
        return equ

    def setLevel(self):
        l = toInt(
            '\nSelecione o nível de dificuldade\n\n'
            '0 -> Fácil\n'
            '1 -> Médio\n'
            '2 -> Difícil\n'
            '\n> '
        )
        if l not in self.difficult:
            print('Nível inválido, tente novamente.')
            return self.setLevel()
        print('Nível selecionado: %s' % {0: 'fácil', 1: 'médio', 2: 'difícil'}[l])
        self.level = l

    def doQuestion(self):
        qu = self.genEq()
        print('\nQuestão [%d]' % self.question, end='\n\n')
        print('Quanto é %s?' % qu, end='\n\n')
        asw = toInt('> ')
        if asw != eval(qu): # função eval calcula uma string com expressão matemática
            if not confirm('Poxa... resposta incorreta. Deseja continuar? [S/N] > '):
                exit(self.end())
            self.question += 1
            return
        print('Resposta correta!')
        c = self.congr.get(self.score)
        if c:
            print('\n[+] ' + c)
        self.score += 1
        self.question += 1

    def end(self):
        clear()
        print('Game over. Score: %d' % self.score)


def toInt(t):
    try:
        return int(input(t))
    except ValueError:
        print('Número inválido.')
    return toInt(t)


def clear():
    system('cls || clear')


confirm = lambda t: input(t).lower() in 'sy'



if __name__ == '__main__':

    game = Game()
    print('=== Arithmetic Game ===')
    print('=[ Para sair, pressione Ctrl+C ]=')

    try:
        game.setLevel()
        while 1:
            game.doQuestion()
    except KeyboardInterrupt:
        game.end()