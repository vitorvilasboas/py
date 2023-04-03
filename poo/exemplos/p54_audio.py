"""**Exemplo**: Abra e reproduza o áudio de um arquivo MP3."""

# pip install pygame

import pygame

pygame.init()
pygame.mixer.music.load('jimgle.mp3')
pygame.mixer.music.play()
pygame.event.wait()