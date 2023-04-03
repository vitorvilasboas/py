import pygame
from pygame.sprite import Sprite

class Al_Bullet(Sprite):
    def __init__(self,ai_s,screen,alien):
        super().__init__()
        self.screen=screen
        self.rect=pygame.Rect(0,0,ai_s.bullet_w,10)
        self.rect.centerx=alien.rect.centerx
        self.rect.bottom=alien.rect.bottom
        self.y=float(self.rect.y)
        self.color=(230,230,0)
        self.speed=ai_s.bullet_speed/10


    def draw_bullet(self):
   
        pygame.draw.rect(self.screen,self.color,self.rect)
       
    def update(self):
        
        self.y+=self.speed
        self.rect.y=self.y
        
