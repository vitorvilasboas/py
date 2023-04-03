# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
from operator import attrgetter
import copy
import random as rd

class Pesos:
    def __init__(self):
        self.okmove = 20
        self.get = 50
        self.kill = 50
        self.win = 50
        
        self.errmove = 2
        self.noget = 2
        self.errshoot = 2
        self.dead = 1
        
class Ambiente:
    def __init__(self):
        self.mapa = np.zeros((4,4), dtype=int)
        self.start = []
        self.wumpus = []
        self.gold = []
        self.hole = []
        self.validMove = []
        self.alfabeto = None
        self.solucaoOtima = None      
        
class Config:
    def __init__(self):
        self.tamanhoPopulacao = 1000
        self.taxaCruzamento   = 0.85
        self.taxaMutacao      = 0.3
        self.tamanhoTorneio   = 2
        self.taxaElitismo     = 0.99 # relacionado a qtd de melhores pais que permanecerão
        self.geracoes         = 1000

        self.minAcoes   = 3
        self.maxAcoes   = 30
        
class Individuo:
    def __init__(self):
        self.cromossomo = []
        self.fitness = 0
        self.visitou = [(0,0)]
        self.morreu = False
        
class AG:
    def cfgAmbiente(self):
        self.amb.alfabeto = ('N','S','L','O','P','W','Z','D','A')
        
        self.amb.mapa[(0,0)] = 10
        self.amb.mapa[(1,2)] = 4
        self.amb.mapa[(2,3)] = 2
        self.amb.mapa[(0,2)] = self.amb.mapa[(1,3)] = self.amb.mapa[(2,1)] = 3
        
        for x in range(0, len(self.amb.mapa)):
            for y in range(0, len(self.amb.mapa)):
                self.amb.validMove.append((x,y))
                if(self.amb.mapa[x,y] == 10):   self.amb.start.append((x,y))
                elif(self.amb.mapa[x,y] == 3):  self.amb.hole.append((x,y))
                elif(self.amb.mapa[x,y] == 4):  self.amb.wumpus.append((x,y))
                elif(self.amb.mapa[x,y] == 2):  self.amb.gold.append((x,y))
    
    def atualizarMelhorGlobal(self):
        self.superIndividuo = copy.deepcopy(max(self.populacao, key = attrgetter('fitness')))
    
    def evaluateMoving(self, fit, individuo):
        localizacao = individuo.visitou[-1]
        cont = 1
        if localizacao in self.amb.hole: 
            fit = 1 # += self.pesos.dead
            individuo.morreu = True
        elif localizacao in self.amb.wumpus: 
            fit = 1 # += self.pesos.dead
            individuo.morreu = True
        elif localizacao in self.amb.validMove: 
            for i in range(len(individuo.visitou) - 1):
                if localizacao == individuo.visitou[i]:
                    cont = cont - 1
            # print(visitados, cont)
            fit += self.pesos.okmove**cont
        else: 
            fit = 1 # += self.pesos.errmove
        return fit
       
    def evaluateShooting(self, fit, destinoTiro):
        if destinoTiro in self.amb.wumpus: 
            fit += self.pesos.kill
            self.amb.wumpus = []
        else: fit += self.pesos.errshoot
        return fit
 
    def objetivo(self, individuo):
        fit = 0;
        for i in range(0, len(individuo.cromossomo)):
            gene = individuo.cromossomo[i]
            genesMov = ('N','S','L','O')
            genesShoot = ('W','Z','D','A') 
            if gene in genesMov:
                if gene == 'N':   mxy = (-1,0) # movimenta o individuo para o Norte (xatual+1,yatual+0)
                elif gene == 'S': mxy = (1,0)   
                elif gene == 'L': mxy = (0,1)
                elif gene == 'O': mxy = (0,-1)
                posAtual = individuo.visitou[-1]
                posFutura = (posAtual[0] + mxy[0], posAtual[1] + mxy[1])
                individuo.visitou.append(posFutura) 
                fit = self.evaluateMoving(fit, individuo)
            elif gene in genesShoot:
                if gene == 'W':   mira = (1,0)
                elif gene == 'Z': mira = (-1,0)
                elif gene == 'D': mira = (0,1)
                elif gene == 'A': mira = (0,-1)
                destinoTiro = ((individuo.visitou[-1][0] + mira[0], individuo.visitou[-1][1] + mira[1]))
                fit = self.evaluateShooting(fit, destinoTiro)
            elif gene == 'P':
                if individuo.visitou[-1] in self.amb.gold: fit += self.pesos.get
                else:  fit += self.pesos.noget
        
        individuo.fitness = fit    
        return individuo 

    def avaliaPopulacao(self, pop):
        for individuo in pop:
            individuo = self.objetivo(individuo)
            print(individuo.fitness, individuo.morreu)
            
    def iniciaPopulacao(self):
        for individuo in self.populacao:
            individuo.cromossomo = np.random.choice(self.amb.alfabeto, np.random.randint(self.cfg.minAcoes, self.cfg.maxAcoes))
            
        self.avaliaPopulacao(self.populacao)
        #self.atualizarMelhorGlobal()
        
    def cruzamento_1_ponto_de_corte(self, p1, p2, f1, f2):
        rand1 = np.random.randint(len(p1.cromossomo))
        rand2 = np.random.randint(len(p2.cromossomo))
        f1.cromossomo = np.concatenate((p1.cromossomo[0:rand1], p2.cromossomo[rand2:len(p2.cromossomo)]))
        f2.cromossomo = np.concatenate((p2.cromossomo[0:rand2], p1.cromossomo[rand1:len(p1.cromossomo)]))
        
    def cruzamento(self):
        for i in range(0, self.cfg.tamanhoPopulacao, 2):
            if self.cfg.taxaCruzamento >= np.random.uniform():
                self.cruzamento_1_ponto_de_corte(self.populacao[self.index[i]], self.populacao[self.index[i+1]], self.filhos[i], self.filhos[i+1])
            else:
                self.filhos[i].cromossomo   = np.copy(self.populacao[self.index[i]].cromossomo)
                self.filhos[i+1].cromossomo = np.copy(self.populacao[self.index[i+1]].cromossomo)
    
    def mutacao(self):
        for individuo in self.filhos:
            teste = np.random.uniform()
            #print(teste)
            if self.cfg.taxaMutacao >= teste:
                
                rand = np.random.randint(len(individuo.cromossomo))
                #print('muta', rand)
                individuo.cromossomo[rand] = np.random.choice(self.amb.alfabeto)
    
    def atualizar(self):
        self.avaliaPopulacao(self.filhos)
        self.populacao.sort(key = attrgetter('fitness'), reverse=True)
        self.filhos.sort(key = attrgetter('fitness'), reverse=True)
        
        for i in range(int(self.cfg.tamanhoPopulacao * self.cfg.taxaElitismo)):
            #print(self.filhos[i].fitness)
            self.populacao[self.cfg.tamanhoPopulacao - (i+1)] = copy.deepcopy(self.filhos[i])
            # self.populacao[i] = copy.deepcopy(self.filhos[self.cfg.tamanhoPopulacao - (i+1)])
    
    
    def __init__(self):
        self.cfg = Config()
        self.amb = Ambiente()
        self.cfgAmbiente()
        self.pesos = Pesos()
        self.individuoOtimo = Individuo()
        self.individuoOtimo.cromossomo = np.asarray(('S','L','D','L','S','L','P','O','N','O','O','N'))
        self.individuoOtimo.fitness = self.objetivo(self.individuoOtimo)
        
        self.populacao       = [Individuo() for i in range(self.cfg.tamanhoPopulacao)]
        self.filhos          = [Individuo() for i in range(self.cfg.tamanhoPopulacao)]
        
        self.superIndividuo  = Individuo()
        self.index           = np.arange(0, self.cfg.tamanhoPopulacao)
        rd.shuffle(self.index) #embaralha lista
        
        self.iniciaPopulacao()
        
        ger = 1
        #for i in range(0, 5):
        while True:
            #print(self.superIndividuo.fitness," - ",self.individuoOtimo.fitness)
            self.cruzamento()
            self.mutacao()
            self.atualizar()
            self.atualizarMelhorGlobal()
            print('Geração %d : fitness: %.2f' % (ger, self.superIndividuo.fitness), self.superIndividuo.cromossomo)
            ger += 1
            if self.superIndividuo.fitness >= self.individuoOtimo.fitness:
                break
            
if __name__ == '__main__':
    #np.random.seed(2)
    AG()


