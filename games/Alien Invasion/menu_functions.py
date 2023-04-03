import sys
import pygame


        
def check_events(stats,buttons,info,click,down,info2) :
    for event in pygame.event.get():
        if event.type == pygame.QUIT :
            sys.exit()
        elif event.type==pygame.MOUSEBUTTONDOWN:
            mouse_x,mouse_y = pygame.mouse.get_pos()
            check_buttons_down(buttons,mouse_x,mouse_y,click,down,info)

        elif event.type==pygame.MOUSEBUTTONUP:
            mouse_x,mouse_y = pygame.mouse.get_pos()
            check_buttons_up(stats,buttons,mouse_x,mouse_y,info,down,info2)

     
            

def update_screen(ai_s,screen,buttons,info,title,down,info2):
    screen.fill(ai_s.bg_c)
    if not info.status and not info2.status:
        title.draw_button()
        for button in buttons :
            button.draw_button()
    if info.status and not info2.status:
        info.draw_screen()
        
        down.draw_button()
        buttons[2].draw_button()
    if info2.status and not info.status:
        info2.draw_screen()
        buttons[2].draw_button()
    pygame.display.flip()
def check_buttons_up(stats,buttons,mouse_x,mouse_y,info,down,info2):
    button=get_button(buttons,mouse_x,mouse_y)
    if button == buttons[0] and not info.status :
        stats.game_active=True
        pygame.mouse.set_visible(False)
        
    if button == buttons[1]:
        info.status = True
        info2.status = False
        
    if button == buttons[2] and not info.status and not info2.status:
        sys.exit()
    if button == buttons[2] and info.status and not info2.status:
        info.status=False
    if button == buttons[2] and info2.status and not info.status:
        info.status= True
        info2.status = False
    if info.status:
        c=down.rect.collidepoint(mouse_x,mouse_y)
        if c:
            info2.status=True
            info.status = False
    
def check_buttons_down(buttons,mouse_x,mouse_y,click,down,info):
    button=get_button(buttons,mouse_x,mouse_y)
    if button:
        click.play()
    c=down.rect.collidepoint(mouse_x,mouse_y)
    if c and info.status:
        click.play()
        
    

def get_button(buttons,mouse_x,mouse_y):
    for button in buttons:
        clicked = button.rect.collidepoint(mouse_x,mouse_y)
        if clicked :
            return button
  
    
    
