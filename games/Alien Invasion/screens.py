import pygame.font
class Info_Screen():
    def __init__(self,screen):
        self.screen = screen
        self.screen_rect = screen.get_rect()
        self.width,self.height =600,600
        self.rect_color = (0,200,0)
        self.text_color = (0,0,0)
        self.font = pygame.font.SysFont(None,30)
        self.title_font = pygame.font.SysFont(None,100)
        self.title = 'Alien Invasion'
        self.subtitle = ' - programmed by Maharshi'
        self.subtitle_font = pygame.font.SysFont(None,25)
        self.rect = pygame.Rect(0,0,self.width,self.height)
        self.rect.center = self.screen_rect.center
        self.msgs=['Defeat the Aliens and earn points by shooting them',
                   'with laser canon. Defeating the aliens brings another',
                   'wave that is more difficult.Beat the HI-Score.',
                   'Use" <-  -> " arrow keys to move horizontally',
                   'Press SPACEBAR to shoot aliens',
                   'Beat the HI-Score',]
       
        self.prep_msgs(self.msgs)
        self.status = False



    def prep_msgs(self,msgs):
        
        self.msg_image_1 = self.font.render(msgs[0],True,self.text_color,self.rect_color)
        self.msg_image_2 = self.font.render(msgs[1],True,self.text_color,self.rect_color)
        self.msg_image_3 = self.font.render(msgs[2],True,self.text_color,self.rect_color)
        self.msg_image_4 = self.font.render(msgs[3],True,self.text_color,self.rect_color)
        self.msg_image_5 = self.font.render(msgs[4],True,self.text_color,self.rect_color)
        
        self.msg_image_rect_1 = self.msg_image_1.get_rect()
        self.msg_image_rect_2 = self.msg_image_2.get_rect()
        self.msg_image_rect_3 = self.msg_image_3.get_rect()
        self.msg_image_rect_4 = self.msg_image_4.get_rect()
        self.msg_image_rect_5 = self.msg_image_5.get_rect()
        
        self.msg_image_rect_1.center=self.rect.center
        self.msg_image_rect_2.center=self.rect.center
        self.msg_image_rect_3.center=self.rect.center
        self.msg_image_rect_4.center=self.rect.center
        self.msg_image_rect_5.center=self.rect.center
           
        self.msg_image_rect_1.centery=self.rect.centery - 100 +50
        self.msg_image_rect_2.centery=self.rect.centery - 50 + 50
        self.msg_image_rect_3.centery=self.rect.centery + 0 + 50
        self.msg_image_rect_4.centery=self.rect.centery + 50 + 50
        self.msg_image_rect_5.centery=self.rect.centery + 100 + 50

        self.title_image = self.title_font.render(self.title,True,self.text_color,self.rect_color)
        self.title_image_rect = self.title_image.get_rect()
        self.title_image_rect.center = self.rect.center
        self.title_image_rect.top = self.rect.top + 20

        self.sub_img = self.subtitle_font.render(self.subtitle,True,self.text_color,self.rect_color)
        self.sub_img_rect = self.sub_img.get_rect()
        self.sub_img_rect.right = self.rect.right - 20
        self.sub_img_rect.bottom = self.title_image_rect.bottom + 20
        


    def draw_screen(self):
        self.screen.fill(self.rect_color,self.rect)
        self.screen.blit(self.title_image,self.title_image_rect)
        self.screen.blit(self.sub_img,self.sub_img_rect)
            
        self.screen.blit(self.msg_image_1,self.msg_image_rect_1)
        self.screen.blit(self.msg_image_2,self.msg_image_rect_2)
        self.screen.blit(self.msg_image_3,self.msg_image_rect_3)
        self.screen.blit(self.msg_image_4,self.msg_image_rect_4)
        self.screen.blit(self.msg_image_5,self.msg_image_rect_5)
        
