import pygame

try:
    pygame.init()
except:
    print("O modulo pygame não foi inicializado com sucesso")

largura=640
altura=480

pygame.display.set_mode((largura,altura))
pygame.display.set_caption("Snake")

sair = False

while not sair:
    for event in pygame.event.get():
        print(event)
        if event.type == pygame.QUIT:
            sair = True
    pygame.display.update()

pygame.quit()

