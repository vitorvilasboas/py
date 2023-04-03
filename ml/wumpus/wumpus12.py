# -*- coding: utf-8 -*-
import numpy as np
from operator import attrgetter
import copy
import random as rd

codigo = 0
contger = 0
        
class Ambiente:
           
    def cfgAmbiente(self):
        self.alfabeto = ('N','S','L','O','P','W','Z','D','A')
        
        self.mapa[(0,0)] = 10
        self.mapa[(1,2)] = 4
        self.mapa[(2,3)] = 2
        self.mapa[(0,2)] = self.mapa[(1,3)] = self.mapa[(2,1)] = 3
        
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
        self.taxaCruzamento   = 0.85
        self.taxaMutacao      = 0.4
        self.tamanhoTorneio   = 2
        self.taxaElitismo     = 0.96 # relacionado a qtd de melhores pais que permanecerão
        self.geracoes         = 1000

        self.minAcoes   = 3
        self.maxAcoes   = 30
        
class Individuo:
    def __init__(self):
        self.id = None
        self.cromossomo = []
        
    def resetAvaliacao(self):
        self.fitness = 0
        self.rotaRetorno = []
        self.visitou = [(0,0)]
        self.mortes = 0
        self.errLeves = 0
        self.errGraves = 0
        self.acertos = 0
        self.pegou = False
        self.matou = False
        self.venceu = False
           
class AG:   
    def atualizarMelhorGlobal(self):
        self.superIndividuo = copy.deepcopy(max(self.populacao, key = attrgetter('fitness')))
    
    def avaliarMovimento1(self, gene, individuo):
        if gene == 'N':   mxy = (-1,0) # movimenta o individuo para o Norte (xatual+1,yatual+0)
        elif gene == 'S': mxy = (1,0)   
        elif gene == 'L': mxy = (0,1)
        elif gene == 'O': mxy = (0,-1)
        local = individuo.visitou[-1]
        local = (local[0] + mxy[0], local[1] + mxy[1])
        if (local in self.amb.hole) or (local in self.amb.wumpus): individuo.mortes += 1
        elif local in self.amb.validMove:
            if local in individuo.visitou: individuo.errLeves += 1 # se ainda nao pegou nem matou retornar é ruim  
            else: individuo.acertos += 1   
        else: individuo.errLeves += 1    
        individuo.visitou.append(local)
        return individuo
    
    def avaliarMovimento2(self, gene, individuo):
        if gene == 'N':   mxy = (-1,0) # movimenta o individuo para o Norte (xatual+1,yatual+0)
        elif gene == 'S': mxy = (1,0)   
        elif gene == 'L': mxy = (0,1)
        elif gene == 'O': mxy = (0,-1)
        local = individuo.visitou[-1]
        local = (local[0] + mxy[0], local[1] + mxy[1])
        if local in self.amb.start: individuo.venceu = True
        else:
            if (local in self.amb.hole) or (local in self.amb.wumpus): individuo.mortes += 1   
            elif local in self.amb.validMove:
                if not individuo.visitou[-1] in individuo.rotaRetorno: individuo.rotaRetorno.append(individuo.visitou[-1])
                if (local in individuo.visitou) and (not(local in individuo.rotaRetorno)):
                    individuo.acertos += 1 # se já pegou e matou retornar é bom  
                    individuo.rotaRetorno.append(local)
                else:
                    individuo.errLeves += 1
            else: individuo.errLeves += 1 
        individuo.visitou.append(local)
        return individuo
    
    def avaliarTiro(self, gene, individuo):
        if gene == 'W':   mira = (1,0)
        elif gene == 'Z': mira = (-1,0)
        elif gene == 'D': mira = (0,1)
        elif gene == 'A': mira = (0,-1)
        destinoTiro = ((individuo.visitou[-1][0] + mira[0], individuo.visitou[-1][1] + mira[1]))
        if destinoTiro in self.amb.wumpus:
            individuo.matou = True
            self.amb.wumpus = []
        else: individuo.errLeves += 1
        return individuo
    
    def avaliarPegar(self, gene, individuo):
        if individuo.visitou[-1] in self.amb.gold:
            individuo.pegou = True
            self.amb.gold = []
        else: individuo.errLeves += 1
        return individuo
    
    def calcularNota(self, In): 
        erros = In.mortes + In.errLeves + In.errGraves
        conquistas = 0
        if In.pegou: conquistas += 1
        if In.matou: conquistas += 1
        if In.venceu: conquistas += 1
        
        penalidades = 1/len(In.visitou)
        
        if erros > 1: fitness = 1
        else: fitness = ((In.acertos*5 + (500*conquistas))) - penalidades
        #fitness = ((acertos + (9*conquistas))**2) - (errLeve + (2*errRegular)) / (errGrave**100)     

        return fitness
    
    def avaliarIndividuo(self, individuo):
        individuo.resetAvaliacao()
        for i in range(0, len(individuo.cromossomo)):
            gene = individuo.cromossomo[i]
            genesMov = ('N','S','L','O')
            genesShoot = ('W','Z','D','A')
            if gene == 'P':
                if individuo.pegou: individuo.errGraves += 1   
                else: individuo = self.avaliarPegar(gene, individuo) 
            elif gene in genesShoot:
                if individuo.matou: individuo.errGraves += 1
                else: individuo = self.avaliarTiro(gene, individuo)      
            elif gene in genesMov:
                if individuo.pegou and individuo.matou: individuo = self.avaliarMovimento2(gene, individuo)
                else: individuo = self.avaliarMovimento1(gene, individuo)          
        individuo.fitness = self.calcularNota(individuo) 
        return individuo 

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
        self.individuoOtimo.cromossomo = np.asarray(('S','L','D','L','S','L','P','O','N','O','O','N'))
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
            print('G%d f:%.2f (%.2f)' % (contger, self.superIndividuo.fitness, self.individuoOtimo.fitness))
            print(self.superIndividuo.cromossomo, '\n')
            if self.superIndividuo.fitness >= self.individuoOtimo.fitness:
                break
          
if __name__ == '__main__':
    #np.random.seed(2)
    AG()