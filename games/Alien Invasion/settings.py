class Settings :
    def __init__(self):
        self.scr_w=800
        self.scr_h=600
        self.bg_c=(0,0,0)
       
        self.ship_limit=1
        self.bullet_color_org = (230,230,230)
        self.bullet_color = self.bullet_color_org
        self.bullet_w_org=5
        self.bullet_w=self.bullet_w_org
        self.bullet_h=5
        self.bullets_allowed=1
               
        self.fleet_drop_speed=20
        self.im = 0
        
        self.speedup_scale=1.3
        self.score_scale=1.5
        self.fleet_direction=1
        self.bullet_health = 0
        self.initialize_dynamic_settings()
    def initialize_dynamic_settings(self):
        self.ts = 40
        self.td = 0
        self.ship_speed=10
        self.bullet_speed=20
        self.alien_points=1
        self.alien_speed=0
        
        
    def increase_speed(self):
        self.ship_speed*=(self.speedup_scale-0.25)
        
        
        self.ts/=self.speedup_scale
        self.alien_points=int(self.alien_points*self.score_scale)
        

