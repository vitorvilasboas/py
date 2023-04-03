import pygame
from pygame.sprite import Sprite

class Bullet(Sprite):
    def __init__(self,ai_s,screen,ship,x,c):
        super().__init__()
        self.screen=screen
        self.c=c
        self.inclination = x
        self.rect=pygame.Rect(0,0,ai_s.bullet_w,ai_s.bullet_h)
        self.rectm=self.rect
        if self.c==0:
            self.rect.centerx=ship.rect.centerx + x
            self.rect.top=ship.rect.top
            
            self.y=float(self.rect.y)
            self.x=float(self.rect.x)
        if self.c==1:
            self.rectm.centerx=ship.mirror_rect.centerx + x
            self.rectm.top=ship.mirror_rect.top
            self.ym=float(self.rectm.y)
            self.xm=float(self.rectm.x)
        self.color=ai_s.bullet_color
        self.speed=ai_s.bullet_speed
        self.health=ai_s.bullet_health


    def draw_bullet(self):
        if self.c==0:
            pygame.draw.rect(self.screen,self.color,self.rect)
        if self.c==1:
            pygame.draw.rect(self.screen,self.color,self.rectm)

    def update(self):
        if self.c==0:
            self.y-=self.speed
            self.rect.y=self.y
            self.x-=(self.speed/2)*self.inclination
            self.rect.x=self.x
        if self.c==1:
            self.ym-=self.speed
            self.rectm.y=self.ym
            self.xm-=(self.speed/2)*self.inclination
            self.rectm.x=self.xm

        
        
        
        
