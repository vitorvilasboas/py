# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas

import copy
import numpy as np
from random import random
from operator import attrgetter

contger = 0

class Ambiente:
    def __init__(self):
        self.pos_largada = (0,0)
        self.loc_wumpus = []
        self.loc_ouro = []
        self.loc_buracos = []
        self.loc_validos = []
        self.alfabeto = ('N','S','L','O','P','^','v','>','<')
        self.minAcoes = 6
        self.maxAcoes = 20
        self.mapa = np.zeros((4,4), dtype=int)
        self.mapa[(0,0)] = 10
        
        self.configMapaFixo()
        
        for x in range(0, len(self.mapa)):
            for y in range(0, len(self.mapa)):
                self.loc_validos.append((x,y))
                if(self.mapa[x,y] == 3):  self.loc_buracos.append((x,y))
                elif(self.mapa[x,y] == 4):  self.loc_wumpus = (x,y)
                elif(self.mapa[x,y] == 2):  self.loc_ouro = (x,y)
    
    def configMapaFixo(self): 
        self.mapa[(1,2)] = 4
        self.mapa[(2,3)] = 2
        self.mapa[(0,2)] = self.mapa[(1,3)] = self.mapa[(2,1)] = 3
        
    def configMapaDinamico(self):
        i = 0
        while i < 3:# # add 3 poços em casas aleatórias
            ln = np.random.randint(4)
            cl = np.random.randint(4)
            if self.mapa[(ln,cl)] == 0: 
                self.mapa[(ln,cl)] = 3
                i+=1
        while True: 
            ln = np.random.randint(4)
            cl = np.random.randint(4)
            if self.mapa[(ln,cl)] == 0:
                self.mapa[(ln,cl)] = 2 # add ouro em casa aleatória
                break
        while True:
            ln = np.random.randint(4)
            cl = np.random.randint(4)
            if self.mapa[(ln,cl)] == 0:
                self.mapa[(ln,cl)] = 4 # add wumpus em casa aleatória
                break

class Individuo:
    def __init__(self, geracao=0):
        self.geracao = geracao
        self.cromossomo = []
        self.trajetoria = [(0,0)]
        self.rotaRetorno = []
        self.nota = 0
        self.penalidades = 0
        self.erroFatal = 0
        self.acertos = 0
        self.pegou = False
        self.matou = False
        self.venceu = False
        self.conquistas = 0
        self.cenario = Ambiente()
    
    def avaliaTiro(self, gene):
        if self.matou: self.penalidades += 5 # se já matou, pq atirar?
        else:
            if gene == '^':   mira = (-1,0)
            elif gene == 'v': mira = (1,0)
            elif gene == '>': mira = (0,1)
            elif gene == '<': mira = (0,-1)
            destinoTiro = ((self.trajetoria[-1][0] + mira[0], self.trajetoria[-1][1] + mira[1]))
            if destinoTiro == self.cenario.loc_wumpus: self.matou = True
            else: self.penalidades += 5 # errou tiro
    
    def avaliaPegar(self, gene):
        if self.pegou: self.penalidades += 5 # se já pegou, pq tentar pegar?
        else:
            if self.trajetoria[-1] == self.cenario.loc_ouro: self.pegou = True
            else: self.penalidades += 5 # não pegou
    
    def avaliaMovimento(self, gene):
        if gene == 'N':   mxy = (-1,0) # movimenta o individuo para o Norte (xatual+1,yatual+0)
        elif gene == 'S': mxy = (1,0)   
        elif gene == 'L': mxy = (0,1)
        elif gene == 'O': mxy = (0,-1)
        local = self.trajetoria[-1]
        local = (local[0] + mxy[0], local[1] + mxy[1])
        if local in self.cenario.loc_buracos: self.erroFatal += 1
        elif (local == self.cenario.loc_wumpus) and (not self.matou): self.erroFatal += 1
        elif local in self.cenario.loc_validos:
            self.acertos += 1
            if self.pegou and self.matou: # se já pegou e matou retornar é bom
                if local in self.trajetoria:
                    self.acertos += 3/(float(local[0]+1)) + 3/(float(local[1]+1)) # ou int(3-local[0] + 3-local[1]))       
                    if not (local in self.rotaRetorno):
                        self.acertos += 4
                        # print("Mutando, tentando voltar...")
                        cont = 0
                        for l in self.trajetoria: 
                            if local == l: cont += 1 #vrf qtas vezes olocal foi vizitado (qto menos passou, mais pontos ganha)
                        if (cont <= 2): self.acertos += 5 # 
                        #else: self.penalidades += (cont**2)
                        self.rotaRetorno.append(local)
                    #else: self.penalidades += 3    
                else: self.penalidades += 3 # Voltando por rota desconhecida, onde ainda não visitou (vagando/vicio)
                if local == self.cenario.pos_largada: self.venceu = True
            else: # se ainda nao pegou nem matou retornar é ruim
                if local in self.trajetoria: 
                    self.penalidades += 2 # Voltando por onde já passou (não explora/vicio)
                else: self.acertos += 3 # explorando (se muito alto pode viciar)
        else: self.erroFatal += 1 #Graves # saiu do mapa
        self.trajetoria.append(local) 
    
    def calculaNota(self):
        if self.pegou: self.conquistas += 1
        if self.matou: self.conquistas += 1
        if self.venceu: self.conquistas += 1
        if self.erroFatal > 0: self.nota = len(self.trajetoria)/float(self.erroFatal + self.penalidades)
        else:
            self.penalidades += 2*(len(self.trajetoria) - 2)
            self.nota = (20*self.conquistas) + self.acertos - self.penalidades
            if self.nota <= 0: self.nota = len(self.trajetoria)/float(self.penalidades)
        
    def avaliacao(self):
        for i in range(len(self.cromossomo)):
            gene = self.cromossomo[i]
            if gene == 'P': self.avaliaPegar(gene) 
            elif gene in ('^','v','>','<'): self.avaliaTiro(gene)      
            elif gene in ('N','S','L','O'): self.avaliaMovimento(gene)         
        self.calculaNota()
            
    def mutacao(self, taxa_mutacao):
        for i in range(len(self.cromossomo)):
            if random() < taxa_mutacao:
                pos = np.random.randint(len(self.cromossomo))
                self.cromossomo[pos] = np.random.choice(self.cenario.alfabeto)
        return self
        
class AlgoritmoGenetico:
    def iniciaPopulacao(self):
        self.populacao = [Individuo() for i in range(self.tamanhoPopulacao)]
        for individuo in self.populacao:
            individuo.cromossomo = np.random.choice(self.cenario.alfabeto, np.random.randint(self.cenario.minAcoes, self.cenario.maxAcoes))
            individuo.cenario = self.cenario
            individuo.avaliacao()
        self.ordenaPopulacao()
        self.melhorGlobal = self.populacao[0]      
        return self.populacao
    
    def ordenaPopulacao(self):
        self.populacao = sorted(self.populacao, key = lambda populacao: populacao.nota, reverse = True)
    
    def atualizaMelhorGlobal(self, individuo):
        #self.melhorGlobal = copy.deepcopy(max(self.populacao, key = attrgetter('nota')))
        if individuo.nota > self.melhorGlobal.nota: self.melhorGlobal = individuo
        
    def selecao(self, soma_notas): # método roleta viciada
        pai = -1
        soma = 0
        i = 0
        limiar = random() * soma_notas # np.random.randint(soma_notas) 
        while i < len(self.populacao) and soma < limiar:
            soma += self.populacao[i].nota
            pai += 1
            i += 1
        return self.populacao[pai] 
     
    def corte1x(self, pais):
        corte1 = np.random.randint(len(pais[0].cromossomo)-1)
        corte2 = np.random.randint(len(pais[1].cromossomo)-1)
        filho1 = np.concatenate((pais[0].cromossomo[0:corte1], pais[1].cromossomo[corte2::]))
        filho2 = np.concatenate((pais[1].cromossomo[0:corte2], pais[0].cromossomo[corte1::]))
        return filho1, filho2
    
    def corte2x(self, pais):
        corte1a = np.random.randint(len(pais[0].cromossomo)-1)
        corte1b = np.random.randint(corte1a, len(pais[0].cromossomo)-1)
        corte2a = np.random.randint(len(pais[1].cromossomo)-1)
        corte2b = np.random.randint(corte2a, len(pais[1].cromossomo)-1)
        filho1 = np.concatenate((pais[0].cromossomo[0:corte1a], pais[1].cromossomo[corte2a:corte2b], pais[0].cromossomo[corte1b::]))
        filho2 = np.concatenate((pais[1].cromossomo[0:corte2a], pais[0].cromossomo[corte1a:corte1b], pais[1].cromossomo[corte2b::]))
        return filho1, filho2
    
    def crossover(self, pais):
        if self.taxaCruzamento >= np.random.uniform():
            filho1, filho2 = self.corte2x(pais)
        else:
            filho1 = np.copy(pais[0].cromossomo)
            filho2 = np.copy(pais[1].cromossomo)
        filhos = [Individuo(self.cont_geracoes + 1), Individuo(self.cont_geracoes + 1)]
        filhos[0].cromossomo = filho1
        filhos[1].cromossomo = filho2     
        return filhos
    
    def atualizarPopulacao(self, filhos):
        for individuo in filhos: individuo.avaliacao()
        filhos = sorted(filhos, key = lambda filhos: filhos.nota, reverse = True)
        for i in range(int(self.tamanhoPopulacao * self.taxaElitismo)):
            self.populacao[self.tamanhoPopulacao - (i+1)] = copy.deepcopy(filhos[i])
        self.ordenaPopulacao()
            
    def __init__(self):
        global contger
        self.n_geracoes       = 1000
        self.tamanhoPopulacao = 100
        self.taxaCruzamento   = 0.85
        self.taxaMutacao      = 0.05 # 1% ou 5%
        self.taxaElitismo     = 0.96 # controla qtd d melhores pais q vingarão
        self.cont_geracoes    = 0
        self.populacao        = []
        self.melhorGlobal     = 0
        self.cenario = Ambiente()
        # testandoo...
        self.individuoOtimo = Individuo()
        self.individuoOtimo.cromossomo = np.asarray(('S','L','>','L','S','L','P','O','N','O','O','N'))
        self.individuoOtimo.avaliacao()
        print(self.cenario.mapa)
        print('%s ' % (self.individuoOtimo.cromossomo))
        #print('Trajetória: %s' % self.individuoOtimo.trajetoria)
        print('EFatal:%s ; Ac:%s ; Pena:%s ; Conq:%s ; (%s)' % (self.individuoOtimo.erroFatal,
                self.individuoOtimo.acertos, self.individuoOtimo.penalidades,
                self.individuoOtimo.conquistas, self.individuoOtimo.nota))
        
        self.iniciaPopulacao()
        '''
        for individuo in self.populacao:
            print('---------------------------------')
            print('%s ' % individuo.cromossomo)
            #print('Trajetória: %s' % individuo.trajetoria)
            print('EFatal:%s ; Ac:%s ; Pena:%s ; Conq:%s ; (%s)' % (individuo.erroFatal,
                    individuo.acertos, individuo.penalidades, individuo.conquistas, individuo.nota))
        '''
        print("G0 -> %s %s" % (self.populacao[0].cromossomo, self.populacao[0].nota))
        #for i in range(self.n_geracoes):
        while True:
            contger += 1
            somaNotas = 0
            for individuo in self.populacao: somaNotas += individuo.nota
            nova_geracao = []
            for indg in range(0, self.tamanhoPopulacao, 2):
                pais = [self.selecao(somaNotas), self.selecao(somaNotas)]
                filhos = self.crossover(pais)
                nova_geracao.append(filhos[0].mutacao(self.taxaMutacao))
                nova_geracao.append(filhos[1].mutacao(self.taxaMutacao))
            self.atualizarPopulacao(list(nova_geracao))
            self.atualizaMelhorGlobal(self.populacao[0])
            print("G%s -> %s %s" % (contger, self.populacao[0].cromossomo, round(self.populacao[0].nota))) 
            if self.melhorGlobal.matou and self.melhorGlobal.pegou and self.melhorGlobal.venceu:
                break
        print(self.cenario.mapa)
        print("\nMELHOR: G%s %s %s " % (contger, self.melhorGlobal.cromossomo, round(self.melhorGlobal.nota)))
        
        
if __name__ == '__main__':
    AlgoritmoGenetico()
    

