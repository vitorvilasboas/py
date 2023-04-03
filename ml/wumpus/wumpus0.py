# -*- coding: utf-8 -*-
from random import random, randrange, choice
from numpy import zeros, concatenate

def criar_ambiente():
    AMB = zeros((4,4), dtype=int)
    
    
    i = 0
    while i < 3:
        linha = randrange(0, 4)
        coluna = randrange(0, 4)
        if (linha!=0 and coluna!=0 and AMB[(linha,coluna)] == 0):
            AMB[(linha,coluna)] = 3
            i += 1

    ouro = True
    while ouro:
        linha = randrange(0, 4)
        coluna = randrange(0, 4)
        if (linha != 0 and coluna != 0 and AMB[(linha, coluna)] == 0):
            AMB[(linha, coluna)] = 2
            ouro = False

    wumpus = True
    while wumpus:
        linha = randrange(0, 4)
        coluna = randrange(0, 4)
        if (linha != 0 and coluna != 0 and AMB[(linha, coluna)] == 0):
            AMB[(linha, coluna)] = 4
            wumpus = False

    return AMB


# Criando primeira geração
def cria_geracao1(alfabeto):
    qtd = randrange(2,10)
    G = []
    for q in range(0, qtd):
        tam = randrange(2, 10)
        Ind = []
        for t in range(0, tam):
            # Ind.append(str ( alfabeto[ randrange( 0, len(alfabeto) ) ] ))
            Ind.append(choice(alfabeto))
        G.append(Ind)
    return G


def mutacao(I, alfabeto):
    I[randrange(0, len(I))] = choice(alfabeto)
    #altera um genes aleatório do individuo, de acordo com o alfabeto
    return I


def crossover(G, alfabeto,AMB):
    qi = len(G);
    if (qi%2==1):
        qi += 1
    qi = qi/2
    i = 0
    while i < qi:
        idx = randrange(0, len(G))
        idy = randrange(0, len(G))
        x = G[idx]
        corte1x = randrange(1,len(x))
        corte2x = randrange(corte1x,len(x))
        y = G[idy]
        corte1y = randrange(1, len(y))
        corte2y = randrange(corte1y, len(y))
        xy = concatenate([x[0:corte1x], y[corte1y:corte2y], x[corte2x:len(x)]])
        yx = concatenate([y[0:corte1y], x[corte1x:corte2x], y[corte2y:len(x)]])
        
        
        # print 'x= ', x
        # print 'y= ', y
        # print corte1x, corte2x, corte1y, corte2y
        # print 'xy= ', xy,'| yx= ', yx

        A = mutacao(xy, alfabeto)
        B = mutacao(yx, alfabeto)

    
        
        # print 'A = ', A
        # print 'B = ', B
    
        i += 1
    fitness(G[qi],AMB)

def fitness(I,AMB):
    cair_buraco = -10
    wumpus = -10
    npegar = -5
    atirar_err = -3
    sair = -2
    mover = -1
    pegar = 5
    matar = 3
    voltar = 2
    posicao_atual = [0,0]
    print len(posicao_atual)
    """
    for gen in range(0, len(I)):
        switch(I[gen]):
            case 'N':
    """
    nx = 1
    ny = 0
    if posicao_atual[0]<0:
        movimenta(AMB, posicao_atual, posicao_atual[0] += nx)
    else 
    
    if posicao_atual[1] < 2 #len(AMB[0]-2):
        movimentar()
        
        
    
    print posicao_atual
    
    # movimentar(AMB,nx,ny)
                  
    
   
    
    
"""
def movimentar(AMB,nx,ny):
    conti = 0
    print len(AMB), len (AMB[0])
    atualizado = False
    while conti < len(AMB):
        contj = 0
        while contj < len (AMB[0]):
            if AMB[conti][contj] >= 10 and not atualizado:
                # pos_atual = (conti,contj)
                print AMB
                AMB[conti][contj] -= 10
                AMB[conti+nx][contj+ny] += 10
                atualizado = True
            contj += 1
        conti += 1
    print(AMB)
    
    return AMB
"""




if __name__ == "__main__":
    alfabeto = ('N', 'S', 'L', 'O', 'P', 'AS', 'AN', 'AL', 'AO')
    alfabeto = ('N', 'S', 'L', 'O', 'P', 'AS', 'AN', 'AL', 'AO')
    AMB = criar_ambiente()
    AMB[(0,0)] = 10
    print(AMB)
    G = cria_geracao1(alfabeto)
    print(G)
    crossover(G, alfabeto, AMB)