import pygame.font
from pygame.sprite import Group
from ship import Ship

class Scoreboard():
    def __init__(self,ai_s,screen,stats):
        self.screen=screen
        self.screen_rect=screen.get_rect()
        self.ai_s=ai_s
        self.stats=stats
        self.power_status = False
        self.power_type = 0

        self.text_color=(230,230,0)
        self.font=pygame.font.SysFont(None,30)
        self.count = 0
        

        self.prep_score()
        
        self.prep_level()
        self.prep_ships()
        self.prep_high_score_text()
        self.prep_high_score()

    def prep_score(self):
       
        rounded_score=int(round(self.stats.score,-1))
        score="{:,}".format(rounded_score)
        score_str=score+"  Score"
        self.score_image=self.font.render(score_str,True,self.text_color,self.ai_s.bg_c)
        self.score_rect=self.score_image.get_rect()
        self.score_rect.right=self.screen_rect.right-20
        self.score_rect.top=20
        
    def show_score(self):
        self.screen.blit(self.score_image,self.score_rect)
        self.screen.blit(self.high_score_image,self.high_score_rect)
        self.screen.blit(self.level_image,self.level_rect)
        self.screen.blit(self.high_score_text_image,self.high_rect)
        self.ships.draw(self.screen)
    def prep_high_score(self):
        high_score=int(round(self.stats.high_score,-1))
        high_score_str="{:,}".format(high_score)
        self.high_score_image=self.font.render(high_score_str,True,self.text_color,self.ai_s.bg_c)
        self.high_score_rect=self.high_score_image.get_rect()
        self.high_score_rect.centerx=self.screen_rect.centerx
        self.high_score_rect.bottom=self.level_rect.bottom
    def prep_high_score_text(self):
        high_score_text_str="HI-Score"
        self.high_score_text_image=self.font.render(high_score_text_str,True,self.text_color,self.ai_s.bg_c)
        self.high_rect=self.high_score_text_image.get_rect()
        self.high_rect.centerx=self.screen_rect.centerx
        self.high_rect.top=self.score_rect.top
    
    def prep_level(self):
        level_no=str(self.stats.level)
        level_str=level_no+' Lvl'
        self.level_image=self.font.render(level_str,True,self.text_color,self.ai_s.bg_c)
        self.level_rect=self.level_image.get_rect()
        self.level_rect.right=self.score_rect.right
        self.level_rect.top=self.score_rect.bottom+10
    def prep_ships(self):
        self.ships=Group()
        for ship_number in range(self.stats.ships_left):
            ship=Ship(self.ai_s,self.screen)
            ship.rect.x=10+ship_number*(ship.rect.width+10)
            ship.rect.y=10
            self.ships.add(ship)
    def count_down(self,power):
        self.count+=1
        if self.count >= 600:
            power.stop()
            self.power_status = False
            self.count = 0
       
            

        
        
