import pygame
import random
class Powers():
    def __init__(self,screen):
        self.screen = screen
        self.screen_rect=screen.get_rect()
        self.rect = pygame.Rect(0,0,50,15)
        
        self.speed = 5
        self.status = False
    def generate_rand_pow(self,pos):
        self.pow = random.randint(1,4)
        if self.pow == 1:
            self.color = (230,0,0)
        if self.pow == 2:
            self.color = (0,230,230)
        if self.pow == 3:
            self.color = (230,230,230)
        if self.pow==4:
            self.color = (127,127,127)
        self.rect.center = pos.center
        self.y = float(self.rect.y)
        
    def draw_power(self):
        pygame.draw.rect(self.screen,self.color,self.rect)
    def update(self):
        self.y += self.speed
        self.rect.y = self.y
        if self.rect.top >= self.screen_rect.bottom :
            self.status = False
        
