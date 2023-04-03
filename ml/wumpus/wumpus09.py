# -*- coding: utf-8 -*-
import numpy as np
from operator import attrgetter
import copy
import random as rd

codigo = 0
contger = 0

class Pesos:
    def __init__(self):
        self.okmove = 5
        self.get = 50
        self.kill = 50
        self.win = 50
        
        self.regress = 1
        self.errmove = 1
        self.noget = 1
        self.errshoot = 1
        self.dead = 1
        
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
        self.taxaCruzamento   = 0.90
        self.taxaMutacao      = 0.3
        self.tamanhoTorneio   = 2
        self.taxaElitismo     = 0.90 # relacionado a qtd de melhores pais que permanecerão
        self.geracoes         = 1000

        self.minAcoes   = 3
        self.maxAcoes   = 30
        
class Individuo:
    def __init__(self):
        self.id = None
        self.cromossomo = []
        
        self.fitness = None
        self.visitou = None
        self.morreu = None
        self.errouTiro = None
        self.naopegou = None
        self.retornou = None
        self.vagando = None
        self.saiu = None
        self.moveuok = None
        self.pegou = False
        self.matou = False
        self.venceu = False
       
        
    def resetAvaliacao(self):
        self.fitness = 0
        self.visitou = [(0,0)]
        self.morreu = 0
        self.errouTiro = 0
        self.naopegou = 0
        self.retornou = 0
        self.vagando = 0
        self.saiu = 0
        self.moveuok = 0
        self.pegou = False
        self.matou = False
        self.venceu = False
           
class AG:   
    def atualizarMelhorGlobal(self):
        self.superIndividuo = copy.deepcopy(max(self.populacao, key = attrgetter('fitness')))
    
    def evaluateMoving(self, individuo):
        localizacao = individuo.visitou[-1]
        retorno = False
        if localizacao in self.amb.hole: 
            individuo.fitness += self.pesos.dead
            individuo.morreu += 1
        elif localizacao in self.amb.wumpus:
            #print('dead', localizacao, individuo.cromossomo)
            #print(individuo.visitou)
            individuo.fitness += self.pesos.dead
            individuo.morreu += 1
        elif localizacao in self.amb.validMove:
            for i in range(len(individuo.visitou) - 1):
                if localizacao == individuo.visitou[i]: retorno = True   
            if individuo.pegou and individuo.matou: # se já pegou e já matou retornar é bom       
                
                if retorno: 
                    individuo.moveuok += 1
                    individuo.fitness += self.pesos.okmove
                else:
                    individuo.vagando += 1
                    individuo.fitness += self.pesos.regress
                    
            else:  # se ainda nao pegou nem matou retornar é ruim 
                if retorno: 
                    individuo.retornou += 1
                    individuo.fitness += self.pesos.regress
                else:
                    individuo.moveuok += 1
                    individuo.fitness += self.pesos.okmove 
            # print(visitados, cont)
            
        else: 
            individuo.fitness = self.pesos.errmove
            individuo.saiu += 1
        return individuo
       
    def evaluateShooting(self, individuo, destinoTiro):
        if destinoTiro in self.amb.wumpus:
            # print('matou', destinoTiro, individuo.cromossomo)
            individuo.fitness += self.pesos.kill
            individuo.matou = True
            self.amb.wumpus = []
        else: 
            individuo.fitness = self.pesos.errshoot
            individuo.errouTiro += 1
        return individuo
 
    def objetivo(self, individuo):
        individuo.resetAvaliacao()
        for i in range(0, len(individuo.cromossomo)):
            gene = individuo.cromossomo[i]
            genesMov = ('N','S','L','O')
            genesShoot = ('W','Z','D','A')
            if individuo.pegou and individuo.matou:
                if gene in genesMov:
                    if gene == 'N':   mxy = (-1,0) # movimenta o individuo para o Norte (xatual+1,yatual+0)
                    elif gene == 'S': mxy = (1,0)   
                    elif gene == 'L': mxy = (0,1)
                    elif gene == 'O': mxy = (0,-1)
                    posAtual = individuo.visitou[-1]
                    posFutura = (posAtual[0] + mxy[0], posAtual[1] + mxy[1])
                    
                    individuo.retornou = 0
                    #print(individuo.retornou)
                    if posFutura in self.amb.start:
                        individuo.venceu = True
                        individuo.fitness += self.pesos.win
                    else:
                        # if posFutura in self.amb.seguros:
                        if posFutura in individuo.visitou:
                            individuo.fitness += self.pesos.okmove
                        else:
                            individuo.vagando += 1
                            individuo.fitness += self.pesos.regress
                    # individuo.visitou.append(posFutura) 
                    # individuo = self.evaluateMoving(individuo)
                elif gene in genesShoot:
                    individuo.fitness = self.pesos.errshoot
                    individuo.errouTiro += 1
                elif gene == 'P':
                    individuo.fitness = self.pesos.noget
                    individuo.naopegou += 1
            else:
                if gene in genesMov:
                    if gene == 'N':   mxy = (-1,0) # movimenta o individuo para o Norte (xatual+1,yatual+0)
                    elif gene == 'S': mxy = (1,0)   
                    elif gene == 'L': mxy = (0,1)
                    elif gene == 'O': mxy = (0,-1)
                    posAtual = individuo.visitou[-1]
                    posFutura = (posAtual[0] + mxy[0], posAtual[1] + mxy[1])
                    individuo.visitou.append(posFutura) 
                    individuo = self.evaluateMoving(individuo)
                elif gene in genesShoot:
                    if gene == 'W':   mira = (1,0)
                    elif gene == 'Z': mira = (-1,0)
                    elif gene == 'D': mira = (0,1)
                    elif gene == 'A': mira = (0,-1)
                    destinoTiro = ((individuo.visitou[-1][0] + mira[0], individuo.visitou[-1][1] + mira[1]))
                    individuo = self.evaluateShooting(individuo, destinoTiro)
                elif gene == 'P':
                    if individuo.visitou[-1] in self.amb.gold:
                        #print('yes', individuo.cromossomo)
                        individuo.fitness += self.pesos.get
                        individuo.pegou = True
                        self.amb.gold = []
                        
                    else:  
                        individuo.fitness = self.pesos.noget
                        individuo.naopegou += 1
        
        if individuo.morreu != 0:
            individuo.fitness = 1
        
        if individuo.saiu != 0:
            individuo.fitness = individuo.fitness/(individuo.saiu+1)
            
        if individuo.retornou != 0:
            individuo.fitness = individuo.fitness/(individuo.retornou+1)
            
        if individuo.vagando != 0:
            #pass
            individuo.fitness = individuo.fitness/(individuo.vagando+1)
            
        if individuo.naopegou != 0:
            individuo.fitness = individuo.fitness/(individuo.naopegou+1)
            
        if individuo.errouTiro != 0:
            individuo.fitness = individuo.fitness/(individuo.errouTiro+1)
            
        individuo.fitness += 1/len(individuo.visitou)
            
        return individuo 

    def avaliaPopulacao(self, pop):
        self.amb.cfgAmbiente()
        for individuo in pop:
            
            individuo = self.objetivo(individuo)        
            
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
        self.avaliaPopulacao(self.populacao)
        #self.atualizarMelhorGlobal()
    
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
        # print(p1.cromossomo,'\n',p2.cromossomo,'\n',f1.cromossomo,'\n',f2.cromossomo)
    
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
        self.avaliaPopulacao(self.filhos)
        self.populacao.sort(key = attrgetter('fitness'), reverse=True)
        self.filhos.sort(key = attrgetter('fitness'), reverse=True)
        for i in range(int(self.cfg.tamanhoPopulacao * self.cfg.taxaElitismo)):
            self.populacao[self.cfg.tamanhoPopulacao - (i+1)] = copy.deepcopy(self.filhos[i])
        
    
    def __init__(self):
        self.cfg = Config()
        self.amb = Ambiente()
        
        self.pesos = Pesos()
        self.individuoOtimo = Individuo()
        self.individuoOtimo.cromossomo = np.asarray(('S','L','D','L','S','L','P','O','N','O','O','N'))
        self.individuoOtimo = self.objetivo(self.individuoOtimo)
        print('fo: ',self.individuoOtimo.fitness)
        
        self.populacao       = [Individuo() for i in range(self.cfg.tamanhoPopulacao)]
        self.filhos          = [Individuo() for i in range(self.cfg.tamanhoPopulacao)]
        
        self.superIndividuo  = Individuo()
        self.index           = np.arange(0, self.cfg.tamanhoPopulacao)
        rd.shuffle(self.index) #embaralha lista
        
        self.iniciaPopulacao()
        
        global contger
        self.atualizarMelhorGlobal()        
        # print('Geração %d : fitness: %.2f' % (contger, self.superIndividuo.fitness), 
        # print(self.superIndividuo.cromossomo, '\n')
        
        #for i in range(0, 3):
        
        while True:
            #print(self.superIndividuo.fitness," - ",self.individuoOtimo.fitness)
            
            #for individuo in self.populacao:
                #print(individuo.id, individuo.fitness)
                #print(individuo.morreu, individuo.cromossomo, individuo.visitou, '\n')
            
            contger += 1
            self.selecao()
            self.cruzamento()
            self.mutacao()
            self.atualizar()
            self.atualizarMelhorGlobal()
            print('G%d f:%.2f' % (contger, self.superIndividuo.fitness))
            print(self.superIndividuo.cromossomo, '\n')
            if self.superIndividuo.fitness >= self.individuoOtimo.fitness:
                break
           
if __name__ == '__main__':
    #np.random.seed(2)
    AG()




