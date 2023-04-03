import pygame
from pygame.sprite import Sprite
class Ship(Sprite):
    def __init__(self,ai_s,screen):
        super().__init__()
        self.screen=screen
        self.image=pygame.image.load('data/ship.png')
        self.mirror_image = pygame.image.load('data/ship_mirror.png')
        self.mirror_rect=self.image.get_rect()
        self.image_red=pygame.image.load('data/ship_red.png')
        self.image_blue=pygame.image.load('data/ship_blue.png')
        self.rect=self.image.get_rect()
       
        self.scr_rect=screen.get_rect()
        self.rect.centerx=self.scr_rect.centerx
        self.rect.bottom=self.scr_rect.bottom
        self.moving_right=False
        self.moving_left=False
        self.ai_s=ai_s
        self.center=float(self.rect.centerx)
        self.fire = False
        
      

    def update(self):
        if self.moving_right and self.rect.right < self.scr_rect.right :
            self.center+=self.ai_s.ship_speed
        if self.moving_left and self.rect.left > 0 :
            self.center-=self.ai_s.ship_speed
        self.rect.centerx=self.center

     
    def blitme(self):
        
        self.screen.blit(self.image,self.rect)
    def blitme_mirror(self):
        self.mirror_rect.center=self.rect.center
        self.mirror_rect.centerx=(self.rect.centerx)*(-1)+self.ai_s.scr_w
        self.screen.blit(self.mirror_image,self.mirror_rect)
        
    def blitme_red(self):
        self.screen.blit(self.image_red,self.rect)
    def blitme_blue(self):
        self.screen.blit(self.image_blue,self.rect)
    def center_ship(self):
        self.center=self.scr_rect.centerx
