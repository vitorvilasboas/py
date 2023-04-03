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
        self.erros = 0
        self.erroFatal = 0
        self.acertos = 0
        self.ganho_cond = 0
        self.perda_cond = 0
        self.pegou = False
        self.matou = False
        self.venceu = False
        self.conquistas = 0
        self.cenario = Ambiente()
    
    def avaliaTiro(self, gene):
        if self.matou: self.erroFatal += 1 # se já matou, pq atirar?
        else:
            if gene == '^':   mira = (-1,0)
            elif gene == 'v': mira = (1,0)
            elif gene == '>': mira = (0,1)
            elif gene == '<': mira = (0,-1)
            destinoTiro = ((self.trajetoria[-1][0] + mira[0], self.trajetoria[-1][1] + mira[1]))
            if destinoTiro == self.cenario.loc_wumpus: self.matou = True
            else: self.erroFatal += 1 # errou tiro
    
    def avaliaPegar(self, gene):
        if self.pegou: self.erroFatal += 1 # se já pegou, pq tentar pegar?
        else:
            if self.trajetoria[-1] == self.cenario.loc_ouro: self.pegou = True
            else: self.erroFatal += 1 # não pegou
    
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
                    self.ganho_cond += 2/float(local[0]+1) + (local[1]+1) # qto mais próximo da largada melhor
                    cont = 0
                    for l in self.trajetoria: 
                        if local == l: cont += 1 #qtas vezes o local atual foi vizitado (qto menos melhor)
                    if cont>2: self.perda_cond += float(cont)**3
                    
                    if local in self.rotaRetorno: self.erros += 1 # já voltou por ai
                    else: self.rotaRetorno.append(local)
                    
                else: self.erros += 1 # Voltando por rota desconhecida, onde ainda não visitou (vagando/vicio)
                
                if local == self.cenario.pos_largada: self.venceu = True
            else: # se ainda nao pegou nem matou retornar é ruim
                if local in self.trajetoria: self.erros += 1 # Voltando por onde já passou (não explora/vicio)
                else: self.acertos += 1 # explorando (se muito alto pode viciar)
        else: self.erroFatal += 1 #Graves # saiu do mapa
        self.trajetoria.append(local) 
    
    def calculaNota(self):
        if self.pegou: self.conquistas += 20
        if self.matou: self.conquistas += 20
        if self.venceu: self.conquistas += 20
        if self.erroFatal > 0: self.nota = 1
        else:
            self.erros = self.erros * 10
            self.acertos = self.acertos * 3
            total_perdas = self.erros + len(self.cromossomo) + self.perda_cond
            total_ganhos = self.conquistas + self.acertos + self.ganho_cond
            self.nota = total_ganhos - total_perdas 
        
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
        limiar = np.random.randint(soma_notas) # round(random()*soma_notas)
        while i < len(self.populacao) and soma < limiar:
            soma += self.populacao[i].nota
            pai += 1
            i += 1
        return self.populacao[pai] 
        
    def crossover(self, pais):
        if self.taxaCruzamento >= np.random.uniform():
            corte1 = np.random.randint(len(pais[0].cromossomo)-1)
            corte2 = np.random.randint(len(pais[1].cromossomo)-1)
            filho1 = np.concatenate((pais[0].cromossomo[0:corte1], pais[1].cromossomo[corte2::]))
            filho2 = np.concatenate((pais[1].cromossomo[0:corte2], pais[0].cromossomo[corte1::]))
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
        self.taxaElitismo     = 0.95 # controla qtd d melhores pais q vingarão
        self.cont_geracoes    = 0
        self.populacao        = []
        self.melhorGlobal     = 0
        self.cenario = Ambiente()
        #self.individuoOtimo = Individuo()
        #self.individuoOtimo.cromossomo = np.asarray(('S','L','>','L','S','L','P','O','N','O','O','N'))
        
        self.iniciaPopulacao()
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
        print("\nMELHOR: G%s %s %s " % (self.melhorGlobal.geracao, self.melhorGlobal.cromossomo,
              round(self.melhorGlobal.nota)))
    
if __name__ == '__main__':
    AlgoritmoGenetico()
    