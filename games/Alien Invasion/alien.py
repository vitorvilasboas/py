import pygame
from pygame.sprite import Sprite
class Alien(Sprite):
    def __init__(self,ai_s,screen):
        super().__init__()
        self.screen=screen
        self.ai_s=ai_s
        self.image_1=pygame.image.load('data/a_1.png')
        self.image_2 =pygame.image.load('data/a_2.png')
        self.image_1_white=pygame.image.load('data/a_1_white.png')
        self.image_2_white =pygame.image.load('data/a_2_white.png')
        self.rect = self.image_1.get_rect()
        self.rect.width=60
        self.rect.height = 44
        self.bonus = False
    def blitme(self):
        if self.ai_s.im == 0:
            self.screen.blit(self.image_1,self.rect)
        else :
            self.screen.blit(self.image_2,self.rect)
    def blitme_white(self):
        if self.ai_s.im == 0:
            self.screen.blit(self.image_1_white,self.rect)
        else :
            self.screen.blit(self.image_2_white,self.rect)
            
            
     
    def update(self):
        self.x+=self.ai_s.alien_speed*self.ai_s.fleet_direction
        self.rect.x=self.x
     
                    
    def check_edges(self):
        screen_rect = self.screen.get_rect()
        if self.rect.right >= screen_rect.right:
            return True
        elif self.rect.left <= 0:
            return True
