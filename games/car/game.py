import pygame
from random import randint
import time

pygame.init()

# parãmetros do carro amarelo
x = 355
y = 100
# parãmetros do carro azul
pos_x_a = 215
pos_y_a = randint(2200, 3000)
# parãmetros do carro de corrida
pos_x_c = 355
pos_y_c = randint(1200, 2000)
# parãmetros do carro branco
pos_x_b = 500
pos_y_b = randint(800, 1000)
# prãmetros da tela de fundo
pos_x_f = 0
pos_y_f = 0
# parametros da moeda
pos_x_m = 335
pos_y_m = -400
# parametros da bruxa
pos_x_brx = -1000
pos_y_brx = 200
# parametros da explosao (apenas para armazenar o valor de forma temporária)
ex = - 500
ey = - 500
# variaveis do tempo
timer = 0
tempo_segundo = 0

# variaveis da velocidade dos carros
velocidade = 10
velocidade_outros = 5
velocidade_moeda = -5

velocidade_b = 5

pontuacao = 0
pontuacao_final = 0

colisao = False

fundo = pygame.image.load('4C9D2B44-E178-42EC-8508-DFF38705A4EB.png')
voce = pygame.image.load('IMG_9997.PNG')
carro_azul = pygame.image.load('IMG_9996.png')
carro_corrida = pygame.image.load('IMG_9998.png')
carro_branco = pygame.image.load('IMG_9999.png')
moeda = pygame.image.load('IMG_0129.png')
game_over = pygame.image.load('2.png')
bruxa = pygame.image.load('IMG_0130.png')
explosao= pygame.image.load('IMG_0131.png')

# criação da janela do timer durante o jogo
font = pygame.font.SysFont('arial black', 30)
texto = font.render("Tempo: ", True, (255, 255, 255), (0, 0, 0))
pos_texto = texto.get_rect()
pos_texto.center = (65, 60)

# criação da janela da pontuação quando apaece a tela de game over
font2 = pygame.font.SysFont('arial black', 30)
texto2 = font2.render("Pontos:", True, (255, 255, 255), (0, 0, 0))
pos_texto2 = texto2.get_rect()
pos_texto2.center = (60, 20)


# criação da janela (painel onde o jogo acontece)
janela = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CRIANDO UM JOGO COM PYTHON DAVID PROGAMAÇÕES")

# adiciona musica de fundo e efeitos sonoros
musica_de_fundo = pygame.mixer.Sound('BoxCat Games - CPU Talk.mp3')
musica_de_fundo.play()
musica_de_fundo.set_volume(0.5)

som_moeda = pygame.mixer.Sound('salamisound-4724461-sfx-collect-gather-2.mp3')
rizada = pygame.mixer.Sound('AUDIO-2022-04-13-16-07-03.mp3')

game_over_m = pygame.mixer.Sound('mixkit-musical-game-over-959.mp3')

# enquanto a janela receber True ela vai ficar sendo exibida para sempre
janela_aberta = True
while janela_aberta:
    pygame.time.delay(10)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            janela_aberta = False

    comandos = pygame.key.get_pressed()

    # contagem de tempo em segundos
    if timer < 23:
        timer += 1
    else:
        tempo_segundo += 1
        texto = font.render("Tempo: " + str(tempo_segundo), True, (255, 255, 255), (0, 0, 0))
        timer = 0
        tempo_final = tempo_segundo

    # Controles do carro amarelo
    if comandos[pygame.K_RIGHT] and x <= 588:
        x += velocidade
    if comandos[pygame.K_LEFT] and x >= 135:
        x -= velocidade
    if comandos[pygame.K_UP]:
        y -= velocidade
    if comandos[pygame.K_DOWN]:
        y += velocidade
    if comandos[pygame.K_1]:
        veocidade_b = 0
    if comandos[pygame.K_2]:
        veocidade_b = 15

        # sorteia uma posição aleatória no eixo y quando o carro sair da tela
        # carro azul
    if pos_y_a <= -180:
        pos_y_a = randint(800, 1000)

        # carro corrida
    if pos_y_c <= -180:
        pos_y_c = randint(1200, 2000)

        # carro branco
    if pos_y_b < -180:
        pos_y_b = randint(2200, 3000)
        # moeda
    if pos_y_m >= 800:
        pos_x_m = randint(180, 540)
        pos_y_m = randint(-1100, -900)

    if pos_x_brx > 7000:
        pos_x_brx = randint(-160000, 100000)

    # Detecta a colisão:
    # carro branco

    if (x - 82 < pos_x_b and y - 175 < pos_y_b) and (x + 82 > pos_x_b and y + 175 > pos_y_b):
        velocidade_outros = 0
        time.sleep(5)
        colisao = True
        musica_de_fundo.stop()

        # carro azul
    if (x - 82 < pos_x_a and y - 175 < pos_y_a) and (x + 82 > pos_x_a and y + 175 > pos_y_a):
        velocidade_outros = 0
        time.sleep(5)
        colisao = True
        musica_de_fundo.stop()

        # carro de corrida
    if (x - 82 < pos_x_c and y - 175 < pos_y_c) and (x + 82 > pos_x_c and y + 175 > pos_y_c):
        velocidade_outros = 0
        time.sleep(0.1)
        colisao = True
        musica_de_fundo.stop()

    # moeda
    if (x - 50 < pos_x_m and y - 40 < pos_y_m) and (x + 82 > pos_x_m and y + 175 > pos_y_m):
        pontuacao += 1
        pontuacao_final = pontuacao
        texto2 = font2.render("Pontos: " + str(pontuacao_final), True, (255, 255, 255), (0, 0, 0))
        pos_y_m = 800
        som_moeda.play()

    # rizada da bruxa
    if pos_x_brx >= 0 and pos_x_brx <= 800:
        rizada.play()

    # tela de game over
    if colisao:
        x = 5000
        pos_x_a = 5000
        pos_y_b = 5000
        pos_y_c = 5000
        pos_x_f = 5000
        pos_x_m = 5000
        pos_y_brx = 5000
        velocidade_b = 0
        pontuacao = pontuacao_final
        som_moeda.stop()
        game_over_m.play()

        # faz a contagem de tempo parar e exibir o tempo em que o usuário ficou no jogo
        pos_texto.center = (540, 520)
        tempo_segundo = tempo_segundo
        tempo_final = tempo_segundo
        texto = font.render("" + str(tempo_final), True, (255, 255, 255), (0, 0, 0))
        timer = 0

        # faz exibir a pontuação do jogo, obtida pela quantidade de moedas coletadas
        pontuacao_final = pontuacao
        pontuacao = pontuacao * 0
        pos_texto2.center = (570, 465)
        texto2 = font2.render("" + str(pontuacao_final), True, (255, 255, 255), (0, 0, 0))

    # Contagem da pontuação
    # Adiciona a velocidade de movimento no eixo y dos carros
    pos_y_a -= velocidade_outros
    pos_y_c -= velocidade_outros + 2
    pos_y_b -= velocidade_outros
    # adiciona movimento contrário na moeda
    pos_y_m -= velocidade_outros - 10

    # adiciona movimento a bruxa
    pos_x_brx += velocidade_b

    # cria as janelas de cada objeto com os seus respetivos parâmetros
    janela.blit(game_over, (0, 0))
    janela.blit(fundo, (pos_x_f, pos_y_f))
    janela.blit(voce, (x, y))
    janela.blit(carro_azul, (pos_x_a, pos_y_a))
    janela.blit(carro_branco, (pos_x_b, pos_y_b))
    janela.blit(carro_corrida, (pos_x_c, pos_y_c))
    janela.blit(moeda, (pos_x_m, pos_y_m))
    janela.blit(texto, pos_texto)
    janela.blit(texto2, pos_texto2)
    janela.blit(bruxa, (pos_x_brx, pos_y_brx))
    pygame.display.update()



# fecha o jogo
pygame.quit()
