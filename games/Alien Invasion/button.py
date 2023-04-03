import pygame.font
class Button():
    def __init__(self,ai_s,screen,msg,width,height,x,y,rc,tc,fs):
        self.screen=screen
        self.screen_rect=screen.get_rect()

        self.width,self.height=width,height
        self.rect_color=rc
        
    
        self.text_color=tc
        self.font=pygame.font.SysFont(None,fs)

        self.rect=pygame.Rect(0,0,self.width,self.height)
        self.rect.centerx = self.screen_rect.centerx + x
        self.rect.centery = self.screen_rect.centery + y
        self.msg=msg

        self.prep_msg(msg)
    def prep_msg(self,msg):

        self.msg_image=self.font.render(msg,True,self.text_color,self.rect_color)
        self.msg_image_rect=self.msg_image.get_rect()
        self.msg_image_rect.center=self.rect.center

    def draw_button(self):
      
     
        self.screen.fill(self.rect_color,self.rect)
        self.screen.blit(self.msg_image,self.msg_image_rect)
    
