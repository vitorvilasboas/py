# -*- coding: utf-8 -*-
#@ author: Vitor M Vilas Boas
import numpy as np
from operator import attrgetter
import copy
import random as rd

codigo = 0
contger = 0
        
class Ambiente:

    def cfgMapaDinamico(self):
        i = 0
        while i < 3
            ln = rd.randrange(0, 4)
            cl = rd.randrange(0, 4)
            if ( (ln != 0 or cl != 0) and (self.mapa[(ln,cl)] == 0) ):
                self.mapa[(ln,cl)] = 3 :# add 3 poços em casas aleatórias
                i+=1
        while True: 
            ln = rd.randrange(0, 4)
            cl = rd.randrange(0, 4)
            if ( (ln != 0 or cl != 0) and (self.mapa[(ln,cl)] == 0) ):
                self.mapa[(ln,cl)] = 2 # add ouro em casa aleatória
                break
        while True:
            ln = rd.randrange(0, 4)
            cl = rd.randrange(0, 4)
            if ( (ln != 0 or cl != 0) and (self.mapa[(ln,cl)] == 0) ):
                self.mapa[(ln,cl)] = 4 # add wumpus em casa aleatória
                break   
     
    def cfgMapaFixo(self):
        self.mapa1 = np.zeros((4,4), dtype=int)
        self.mapa1[(0,0)] = 10
        self.mapa1[(1,2)] = 4
        self.mapa1[(2,3)] = 2
        self.mapa1[(0,2)] = self.mapa1[(1,3)] = self.mapa1[(2,1)] = 3
        
        self.mapa2 = np.zeros((4,4), dtype=int)
        self.mapa2[(0,0)] = 10
        self.mapa2[(3,3)] = 4
        self.mapa2[(2,1)] = 2
        self.mapa2[(0,2)] = self.mapa2[(2,3)] = self.mapa2[(1,2)] = 3
        
        self.mapa3 = np.zeros((4,4), dtype=int)
        self.mapa3[(0,0)] = 10
        self.mapa3[(0,3)] = 4
        self.mapa3[(3,3)] = 2
        self.mapa3[(0,1)] = self.mapa3[(1,1)] = self.mapa3[(2,1)] = 3
        
    def cfgAmbiente(self):
        self.alfabeto = ('N','S','L','O','P','^','v','>','<')
        
        #self.cfgMapaDinamico()
        self.cfgMapaFixo()
        self.mapa = self.mapa1 # definir um dos mapas pré-configurados (fixos)
        
        self.validMove = []
        self.seguros = []
        for x in range(0, len(self.mapa)):
            for y in range(0, len(self.mapa)):
                self.validMove.append((x,y))
                if(self.mapa[x,y] == 10):   self.start.append((x,y))
                elif(self.mapa[x,y] == 3):  self.hole.append((x,y))
                elif(self.mapa[x,y] == 4):  self.wumpus.append((x,y))
                elif(self.mapa[x,y] == 2):  self.gold.append((x,y))
                
        for i in range(len(self.validMove)):
            if self.validMove[i] not in self.hole and self.validMove[i] not in self.wumpus:
                self.seguros.append(self.validMove[i])
     
    def __init__(self):
        self.mapa = np.zeros((4,4), dtype=int)
        self.mapa[(0,0)] = 10
        self.start = []
        self.wumpus = []
        self.gold = []
        self.hole = []
        self.validMove = []
        self.seguros = []
        self.alfabeto = None
        self.solucaoOtima = None
        self.cfgAmbiente()
           
class Config:
    def __init__(self):
        self.tamanhoPopulacao = 100
        self.taxaCruzamento   = 0.95
        self.taxaMutacao      = 0.9
        self.tamanhoTorneio   = 2
        self.taxaElitismo     = 0.95 # controla qtd d melhores pais q vingarão
        self.minAcoes         = 6
        self.maxAcoes         = 20
        
class Individuo:
    def __init__(self):
        self.id = None
        self.cromossomo = []    
    def resetAvaliacao(self):
        self.fitness = 0
        self.rotaRetorno = []
        self.visitou = [(0,0)]
        self.errLeves = 0
        self.errGraves = 0
        self.acertos = 0
        self.pegou = False
        self.matou = False
        self.venceu = False
           
class AG:   
    def atualizarMelhorGlobal(self):
        self.superIndividuo = copy.deepcopy(max(self.populacao, key = attrgetter('fitness')))
    
    def avaliarMovimento(self, gene, In):
        if gene == 'N':   mxy = (-1,0) # movimenta o individuo para o Norte (xatual+1,yatual+0)
        elif gene == 'S': mxy = (1,0)   
        elif gene == 'L': mxy = (0,1)
        elif gene == 'O': mxy = (0,-1)
        local = In.visitou[-1]
        local = (local[0] + mxy[0], local[1] + mxy[1])
        if (local in self.amb.hole) or (local in self.amb.wumpus): In.errGraves += 1
        elif local in self.amb.validMove:
            In.acertos += 1
            if In.pegou and In.matou: # se já pegou e matou retornar é bom
                In.acertos += 1 # acerto só por ser um movimento após pegar e matar
                if local in self.amb.start: In.venceu = True
                ganho = 2/(float(local[0]+1)) + 2/(float(local[1]+1))
                In.acertos += ganho
                if local in In.visitou:# and (not(local in In.rotaRetorno)):
                    print(gene, "Mutando, tentando voltar...")
                    In.acertos += 1 # se já pegou e matou retornar é bom  
                else: In.errLeves += 1
            else: # se ainda nao pegou nem matou retornar é ruim
                if local in In.visitou: In.errLeves += 1
                else: In.acertos += 2   
        else: In.errGraves += 1          
        In.visitou.append(local)
        return In
    
    def avaliarTiro(self, gene, In):
        if In.matou: In.errGraves += 1 # se já matou, pq atirar?
        else:
            if gene == '^':   mira = (-1,0)
            elif gene == 'v': mira = (1,0)
            elif gene == '>': mira = (0,1)
            elif gene == '<': mira = (0,-1)
            destinoTiro = ((In.visitou[-1][0] + mira[0], In.visitou[-1][1] + mira[1]))
            if destinoTiro in self.amb.wumpus:
                In.matou = True
                self.amb.wumpus = []
            else: In.errGraves += 1
        return In
    
    def avaliarPegar(self, gene, In):
        if In.pegou: In.errGraves += 1 # se já pegou, pq tentar pegar?
        else:
            if In.visitou[-1] in self.amb.gold:
                In.pegou = True
                self.amb.gold = []
            else: In.errGraves += 1
        return In
    
    def calcularNota(self, In): 
        conquistas = 0
        if In.pegou: conquistas += 1
        if In.matou: conquistas += 1
        if In.venceu: conquistas += 1
        
        penalidades = len(In.visitou)*2
        #print(In.acertos, conquistas, In.errGraves, In.errLeves, penalidades)
        if In.errGraves > 0: fitness = 1
        else: fitness = (20*conquistas) + In.acertos - penalidades - In.errLeves 
        #fitness = ((acertos + (9*conquistas))**2) - (errLeve + (2*errRegular)) / (errGrave**100)     

        return fitness
    
    def avaliarIndividuo(self, In):
        In.resetAvaliacao()
        for i in range(0, len(In.cromossomo)):
            gene = In.cromossomo[i]
            if gene == 'P': In = self.avaliarPegar(gene, In) 
            elif gene in ('^','v','>','<'): In = self.avaliarTiro(gene, In)      
            elif gene in ('N','S','L','O'): 
                In = self.avaliarMovimento(gene, In)         
        In.fitness = self.calcularNota(In) 
        return In 

    def avaliarPopulacao(self, pop):
        self.amb.cfgAmbiente()
        for individuo in pop:
            individuo = self.avaliarIndividuo(individuo)        
    
    def rotular(self):
        global codigo
        global contger
        codigo += 1
        rotulo = str(contger) + '_' + str(codigo)
        return rotulo
                  
    def iniciaPopulacao(self):
        for individuo in self.populacao:
            individuo.cromossomo = np.random.choice(self.amb.alfabeto, np.random.randint(self.cfg.minAcoes, self.cfg.maxAcoes))
            individuo.id = self.rotular()
        self.avaliarPopulacao(self.populacao)
    
    def selecao(self):       
        for i in range(self.cfg.tamanhoPopulacao):
            tn = np.random.choice(self.cfg.tamanhoPopulacao, self.cfg.tamanhoTorneio, replace = False) # escolhe 2 de 100 sem repetir
            listaTorneio = [self.populacao[tn[i]] for i in range(self.cfg.tamanhoTorneio)]
            self.index[i] = tn[listaTorneio.index(max(listaTorneio, key = attrgetter('fitness')))]

    def cruzamento_corte1x(self, p1, p2, f1, f2):
        rand1 = np.random.randint(len(p1.cromossomo))
        rand2 = np.random.randint(len(p2.cromossomo))
        f1.cromossomo = np.concatenate((p1.cromossomo[0:rand1], p2.cromossomo[rand2:len(p2.cromossomo)]))
        f2.cromossomo = np.concatenate((p2.cromossomo[0:rand2], p1.cromossomo[rand1:len(p1.cromossomo)]))
    
    def cruzamento_corte2x(self, p1, p2, f1, f2):
        rand1a = np.random.randint(len(p1.cromossomo))
        rand1b = np.random.randint(rand1a, len(p1.cromossomo))
        rand2a = np.random.randint(len(p2.cromossomo))
        rand2b = np.random.randint(rand2a, len(p2.cromossomo))
        f1.cromossomo = np.concatenate((p1.cromossomo[0:rand1a], p2.cromossomo[rand2a:rand2b], p1.cromossomo[rand1b:len(p1.cromossomo)]))
        f2.cromossomo = np.concatenate((p2.cromossomo[0:rand2a], p1.cromossomo[rand1a:rand1b], p2.cromossomo[rand2b:len(p2.cromossomo)]))
    
    def cruzamento(self):
        global codigo
        codigo = 99 - int(self.cfg.taxaCruzamento*100)
        for individuo in self.filhos:
            individuo.id = self.rotular()
        for i in range(0, self.cfg.tamanhoPopulacao, 2):
            if self.cfg.taxaCruzamento >= np.random.uniform():
                self.cruzamento_corte2x(self.populacao[self.index[i]], self.populacao[self.index[i+1]], self.filhos[i], self.filhos[i+1])
            else:
                self.filhos[i].cromossomo   = np.copy(self.populacao[self.index[i]].cromossomo)
                self.filhos[i+1].cromossomo = np.copy(self.populacao[self.index[i+1]].cromossomo)
    
    def mutacao(self):
        for individuo in self.filhos:
            if self.cfg.taxaMutacao >= np.random.uniform():
                rand = np.random.randint(len(individuo.cromossomo))
                individuo.cromossomo[rand] = np.random.choice(self.amb.alfabeto)
    
    def atualizar(self):
        self.avaliarPopulacao(self.filhos)
        self.populacao.sort(key = attrgetter('fitness'), reverse=True)
        self.filhos.sort(key = attrgetter('fitness'), reverse=True)
        for i in range(int(self.cfg.tamanhoPopulacao * self.cfg.taxaElitismo)):
            self.populacao[self.cfg.tamanhoPopulacao - (i+1)] = copy.deepcopy(self.filhos[i])
        
    def __init__(self):
        self.cfg = Config()
        self.amb = Ambiente()
        self.individuoOtimo = Individuo()
        self.individuoOtimo.cromossomo = np.asarray(('S','L','>','L','S','L','P','O','N','O','O','N')) # mapa 1
        # self.individuoOtimo.cromossomo = np.asarray(('S','S','L','P','S','L','>','O','N','O','N','N')) # mapa 2
        # self.individuoOtimo.cromossomo = np.asarray(('S','S','S','L','L','L','P','N','N','^','S','S','O','O','O','N','N','N')) # mapa 3
        self.individuoOtimo.id = 'otimo'
        self.individuoOtimo = self.avaliarIndividuo(self.individuoOtimo)
        print('fo:%.2f ' % (self.individuoOtimo.fitness), self.individuoOtimo.cromossomo)
        
        self.populacao       = [Individuo() for i in range(self.cfg.tamanhoPopulacao)]
        self.filhos          = [Individuo() for i in range(self.cfg.tamanhoPopulacao)]
        
        self.superIndividuo  = Individuo()
        self.index           = np.arange(0, self.cfg.tamanhoPopulacao)
        rd.shuffle(self.index) #embaralha lista
        
        self.iniciaPopulacao()
        
        global contger
        self.atualizarMelhorGlobal()        
        while True:
            contger += 1
            self.selecao()
            self.cruzamento()
            self.mutacao()
            self.atualizar()
            self.atualizarMelhorGlobal()
            print('G%d f:%.2f (-%.2f)' % (contger, self.superIndividuo.fitness, 
                  self.individuoOtimo.fitness - self.superIndividuo.fitness), 
                  self.superIndividuo.cromossomo,
                  self.superIndividuo.id)
            
            # flag mapa fixo
            if self.superIndividuo.fitness >= self.individuoOtimo.fitness:   
            # flag mapa dinâmico   
            #if self.superIndividuo.matou and self.superIndividuo.pegou and self.superIndividuo.venceu:
                break
        
        print(self.amb.mapa)
          
if __name__ == '__main__':
    #np.random.seed(2)
    AG()

