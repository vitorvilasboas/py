# -*- coding: utf-8 -*-
import numpy as np
from operator import attrgetter
import copy


"""
    Created on 11/04/2019
    @Author: Demison Rolins 
"""

class Constantes:
    def __init__(self):
        self.tamanhoPopulacao = 100
        self.taxaCruzamento   = 0.85
        self.taxaMutacao      = 0.05
        self.tamanhoTorneio   = 2
        self.taxaElitismo     = 0.95
        self.geracoes         = 1000

        self.limiteSuperior   =  100
        self.limiteInferior   = -100
        self.dimensao         =  30
        
class Individuo:
    def __init__(self):
        self.cromossomo = None
        self.fitness = None

class AG:
    def __init__(self):
        self.cfg = Constantes() #1
        self.populacao       = [Individuo() for i in range(self.cfg.tamanhoPopulacao)] #2
        self.filhos          = [Individuo() for i in range(self.cfg.tamanhoPopulacao)]
        self.superIndividuo  = Individuo()
        self.index           = np.empty(self.cfg.tamanhoPopulacao, dtype = int)
        # print(self.index)

    def atualizarMelhorGlobal(self):
        self.superIndividuo = copy.deepcopy(min(self.populacao, key = attrgetter('fitness')))

    def objetivo(self, cromo):
        return np.sum(cromo**2)

    def avaliarPopulacao(self, pop):
        for individuo in pop:
            individuo.fitness = self.objetivo(individuo.cromossomo)

    def iniciarPopulacao(self): #2
        for individuo in self.populacao:
            individuo.cromossomo = np.random.uniform(self.cfg.limiteInferior, self.cfg.limiteSuperior, self.cfg.dimensao) 
            # print(individuo.cromossomo)
        self.avaliarPopulacao(self.populacao)

    def selecao(self):
        for i in range(self.cfg.tamanhoPopulacao):
            tn = np.random.choice(self.cfg.tamanhoPopulacao, self.cfg.tamanhoTorneio, replace = False) # escolhe 2 de 100 sem repetir
            listaTorneio = [self.populacao[tn[i]] for i in range(self.cfg.tamanhoTorneio)]
            self.index[i] = tn[listaTorneio.index(min(listaTorneio, key = attrgetter('fitness')))]

    def cruzamento_1_ponto_de_corte(self, p1, p2, f1, f2):
        rand = np.random.randint(self.cfg.dimensao)
        f1.cromossomo = np.concatenate((p1.cromossomo[0:rand], p2.cromossomo[rand:len(p2.cromossomo)]))
        f2.cromossomo = np.concatenate((p2.cromossomo[0:rand], p1.cromossomo[rand:len(p1.cromossomo)]))

    def cruzamento(self):
        for i in range(0, self.cfg.tamanhoPopulacao, 2):
            if self.cfg.taxaCruzamento >= np.random.uniform():
                self.cruzamento_1_ponto_de_corte(self.populacao[self.index[i]], self.populacao[self.index[i+1]], self.filhos[i], self.filhos[i+1])
            else:
                self.filhos[i].cromossomo   = np.copy(self.populacao[self.index[i]].cromossomo)
                self.filhos[i+1].cromossomo = np.copy(self.populacao[self.index[i+1]].cromossomo)

    def mutacao(self):
        for individuo in self.filhos:
            if self.cfg.taxaMutacao >= np.random.uniform():
                rand = np.random.randint(self.cfg.dimensao)
                individuo.cromossomo[rand] = np.random.uniform(self.cfg.limiteInferior, self.cfg.limiteSuperior)
                # print(individuo.cromossomo)
    
    def atualizar(self):
        self.avaliarPopulacao(self.filhos)
        self.populacao.sort(key = attrgetter('fitness'), reverse = False) 
        self.filhos.sort(key = attrgetter('fitness'), reverse = False) 

        for i in range(int(self.cfg.tamanhoPopulacao * self.cfg.taxaElitismo)):
            self.populacao[self.cfg.tamanhoPopulacao - (i+1)] = copy.deepcopy(self.filhos[i])

    def opt(self):
        self.iniciarPopulacao()
        for i in range(self.cfg.geracoes):
            self.selecao()
            self.cruzamento()
            self.mutacao()
            self.atualizar()
            self.atualizarMelhorGlobal()
            print('Geração %d : fitness: %f' % (i, self.superIndividuo.fitness))

if __name__ == '__main__':
    np.random.seed(2)
    AG().opt()
