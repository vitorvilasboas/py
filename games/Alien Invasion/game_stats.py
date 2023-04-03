with open('data/high_score.txt','r') as fp:
    line=fp.readline()
hi_score = int(line)
            

class GameStats():
    def __init__(self,ai_s):
        
        self.ai_s=ai_s
        self.reset_stats()
        self.game_active=False
        self.high=hi_score
        self.high_score=self.high
        
        
    def reset_stats(self):
        self.ships_left=self.ai_s.ship_limit
        self.score=0
        self.level=1
    def update_high_score(self):
        if self.high_score > self.high :
            with open('data/high_score.txt','w') as fp:
                fp.write(str(self.high_score))
