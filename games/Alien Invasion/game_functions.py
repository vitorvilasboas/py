import sys
from bullet import Bullet
from alien import Alien
from time import sleep
from al_bullet import Al_Bullet
import pygame
import random

def check_kd_events(event,ship):
    ship.fire = True
    if event.key==pygame.K_d:
    ##if direita>=esquerda:
        ship.moving_right=True
    elif event.key==pygame.K_a:
    ##elif esquerda>=direita:
        ship.moving_left=True        
    elif event.key==pygame.K_q:
        sys.exit()
        
        
                
def check_ku_events(event,ship):
    if event.key==pygame.K_d:
    ##if direita>=esquerda:
        ship.moving_right=False
    elif event.key==pygame.K_a:
    ##elif esquerda>=direita:
        ship.moving_left=False
    elif event.key==pygame.K_SPACE :
        ship.fire = False
    


def check_events(ship) :
    for event in pygame.event.get():
        if event.type == pygame.QUIT :
            sys.exit()
    
            
        elif event.type==pygame.KEYDOWN :
            check_kd_events(event,ship)
        elif event.type==pygame.KEYUP:
            check_ku_events(event,ship)

def update_screen(ai_s,screen,stats,sb,ship,aliens,bullets,po,a_bullets):
    screen.fill(ai_s.bg_c)
    for bull in bullets.sprites():
        bull.draw_bullet()
    
    if bullet_power(sb):
        ship.blitme_red()
    elif bullet_triple(sb):
        ship.blitme_blue()
    elif mirror_power(sb):
        ship.blitme_mirror()
        ship.blitme()
   
    else:
        ship.blitme()
    if freeze_power(sb):
        for ali in aliens:
            ali.blitme_white()
    else:
        for ali in aliens:
            ali.blitme()
    if po.status :
        po.draw_power()
    for b in a_bullets.sprites():
        b.draw_bullet()
    sb.show_score()
    pygame.display.flip()

def update_bullets(ai_s,screen,stats,sb,ship,aliens,bullets,ad,po):
    bullets.update()
    for bullet in bullets.copy():
        if bullet.rect.bottom<=0:
            bullets.remove(bullet)
    check_bullet_alien_collisions(ai_s,screen,stats,sb,ship,bullets,aliens,ad,po)
def alien_bullet(ai_s,screen,aliens,a_bullets,sb,stats):
    if not freeze_power(sb) and stats.game_active:
        if len(a_bullets) == 0 :
            length = len(aliens)
            ran = random.randint(0,length-1)
            i = 0
            for alien in aliens:
                if i == ran:
                    new = Al_Bullet(ai_s,screen,alien)
                    a_bullets.add(new)
                i+=1
def update_a_bullets(ai_s,stats,screen,sb,ship,aliens,bullets,ded,go,game_o,power,a_bullets):
    if stats.game_active :
        a_bullets.update()
        for bullet in a_bullets.copy():
            if bullet.rect.top >= ai_s.scr_h:
                a_bullets.remove(bullet)
        check_b_b_collisions(a_bullets,bullets)
        if pygame.sprite.spritecollideany(ship,a_bullets):
            ship_hit(ai_s,stats,screen,sb,ship,aliens,bullets,ded,go,game_o,power,a_bullets)
            
        
        

def fire_bullet(ai_s,screen,ship,bullets,bs,sb):
    if ship.fire :
        if len(bullets)< ai_s.bullets_allowed:
                
            bs.stop()
            bs.play()
            if bullet_power(sb):
                    
                ai_s.bullet_w=10
                ai_s.bullet_h=10
                ai_s.bullet_color = (255,0,0)
                ai_s.bullet_health = 1
            else :
                ai_s.bullet_w=ai_s.bullet_w_org
                ai_s.bullet_h=ai_s.bullet_w_org
                ai_s.bullet_color = ai_s.bullet_color_org
                ai_s.bullet_health = 0
            if bullet_triple(sb):
                new_bullet = Bullet(ai_s,screen,ship,1,0)
                bullets.add(new_bullet)
                new_bullet = Bullet(ai_s,screen,ship,-1,0)
                bullets.add(new_bullet)
            if mirror_power(sb):
                new_bull = Bullet(ai_s,screen,ship,0,1)
                bullets.add(new_bull)
            new_bullet=Bullet(ai_s,screen,ship,0,0)
            bullets.add(new_bullet)
            
def create_fleet(ai_s,screen,ship,aliens):
    alien=Alien(ai_s,screen)
    number_aliens_x=get_number_aliens_x(ai_s,alien.rect.width)
    number_rows=get_number_rows(ai_s,ship.rect.height,alien.rect.height)-1
    num1=random.randint(0,number_rows-1)
    num2=random.randint(0,number_aliens_x-1)
    for row_number in range(number_rows):
        for alien_number in range(number_aliens_x):
            create_alien(ai_s,screen,aliens,alien_number,row_number,num1,num2)
        
def get_number_aliens_x(ai_s,alien_width):
    available_space_x=ai_s.scr_w-2*alien_width
    number_aliens_x=int(available_space_x/(2*alien_width))
    return number_aliens_x

def create_alien(ai_s,screen,aliens,alien_number,row_number,num1,num2):
        
        alien=Alien(ai_s,screen)
        alien_width=alien.rect.width
        if row_number== num1 and alien_number == num2:
            alien.bonus = True
        alien.x=alien_width+2*alien_width*alien_number
        alien.rect.x=alien.x
        alien.rect.y=alien.rect.height+30+2*alien.rect.height*row_number
        aliens.add(alien)
def get_number_rows(ai_s,ship_height,alien_height):
    available_space_y = (ai_s.scr_h - 5*alien_height) - ship_height
    number_rows=int(available_space_y/(2*alien_height))
    return number_rows
def update_aliens(ai_s,stats,screen,sb,ship,aliens,bullets,ded,go,game_o,power,a_bullets):
    if not freeze_power(sb):
        ai_s.td+=1
        if ai_s.td > ai_s.ts:
            ai_s.im+=1
            if ai_s.im > 1:
                ai_s.im = 0
            ai_s.td=0
            check_fleet_edges(ai_s,aliens)
            aliens.update()
    if pygame.sprite.spritecollideany(ship,aliens):
        ship_hit(ai_s,stats,screen,sb,ship,aliens,bullets,ded,go,game_o,power,a_bullets)
    check_aliens_bottom(ai_s,stats,screen,sb,ship,aliens,bullets,ded,go,game_o,power,a_bullets)
def check_fleet_edges(ai_s,aliens):
    for alien in aliens.sprites():
        if alien.check_edges():
            change_fleet_direction(ai_s,aliens)
            break
def change_fleet_direction(ai_s,aliens):
    for alien in aliens.sprites():
        alien.rect.y+=ai_s.fleet_drop_speed
    ai_s.fleet_direction*=-1
def check_b_b_collisions(a_bullets,bullets):
    for ab in a_bullets:
        for bullet in bullets :
            if pygame.Rect.colliderect(ab.rect,bullet.rect):
                a_bullets.remove(ab)
                bullets.remove(bullet)

def check_bullet_alien_collisions(ai_s,screen,stats,sb,ship,bullets,aliens,ad,po):
    for alien in aliens:
        for bullet in bullets:
            if pygame.Rect.colliderect(alien.rect,bullet.rect):
                if bullet.health == 0 :
                    
                    if alien.bonus :
                        pos=alien.rect
                        po.status = True
                        po.generate_rand_pow(pos)
                    aliens.remove(alien)
                    
                    bullets.remove(bullet)
                    stats.score+=ai_s.alien_points
                    sb.prep_score()
                    check_high_score(stats,sb)
                    ad.stop()
                    ad.play()
                if bullet.health == 1 :
                    if alien.bonus :
                        pos=alien.rect
                        po.status = True
                        po.generate_rand_pow(pos)
                    aliens.remove(alien)
                    bullet.health-=1
                    stats.score+=ai_s.alien_points
                    sb.prep_score()
                    check_high_score(stats,sb)
                    ad.stop()
                    ad.play()
    if len(aliens)==0:
        bullets.empty()
        ai_s.increase_speed()
        stats.level+=1
        sb.prep_level()
        create_fleet(ai_s,screen,ship,aliens)
    
    
    
    
              
                    
    
            


        
def ship_hit(ai_s,stats,screen,sb,ship,aliens,bullets,ded,go,game_o,power,a_bullets):
    if stats.ships_left>0:
        sb.power_status = False
        sb.power_type = 0
        stats.ships_left-=1
        sb.prep_ships()
        aliens.empty()
        bullets.empty()
        a_bullets.empty()
        create_fleet(ai_s,screen,ship,aliens)
        ship.center_ship()
        ded.stop()
        ded.play()
        sleep(0.5)
        power.stop()
    else :
        sb.power_status = False
        power.stop()
        sb.power_type = 0
        ded.stop()
        ded.play()
        sleep(0.5)
        screen.fill(ai_s.bg_c)
        sb.show_score()
        game_o.draw_button()
        pygame.display.flip()
        
        go.play()
        sleep(4)
        stats.game_active=False
        stats.reset_stats()
        sb.prep_score()
        sb.prep_high_score()
        sb.prep_high_score_text()
        sb.prep_level()
        sb.prep_ships()
        stats.update_high_score()

        aliens.empty()
        bullets.empty()
        a_bullets.empty()

        create_fleet(ai_s,screen,ship,aliens)
        ship.center_ship()
        ai_s.initialize_dynamic_settings()
        
        pygame.mouse.set_visible(True)
    

def check_aliens_bottom(ai_s,stats,screen,sb,ship,aliens,bullets,ded,go,game_o,power,a_bullets):
    screen_rect=screen.get_rect()
    for alien in aliens.sprites():
        if alien.rect.bottom >=screen_rect.bottom:
            ship_hit(ai_s,stats,screen,sb,ship,aliens,bullets,ded,go,game_o,power,a_bullets)
            break


def check_high_score(stats,sb):
    if stats.score>stats.high_score:
        stats.high_score=stats.score
        sb.prep_high_score()
def update_po(po,ship,sb,power):
    if check_po_ship_collision(po,ship,sb,power):
        po.status = False
    else:
        po.update()
def check_po_ship_collision(po,ship,sb,power):
    if pygame.Rect.colliderect(po.rect,ship.rect):
        power.stop()
        power.play()
        sb.count=0
        
        
        po.status=False
        sb.power_status=True
        sb.power_type = po.pow
def bullet_power(sb):
    if sb.power_status == True and sb.power_type == 1:
        return True
    else:
        return False
def bullet_triple(sb):
    if sb.power_status == True and sb.power_type == 2:
        return True
    else:
        return False
def freeze_power(sb):
    
    if sb.power_status == True and sb.power_type == 3:
        return True
    else:
        return False
def mirror_power(sb):
    if sb.power_status == True and sb.power_type == 4:
        return True
    else:
        return False
        

    

    
    
