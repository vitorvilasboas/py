# -*- coding: utf-8 -*-
from random import randrange, choice
from numpy import zeros, concatenate

# criação dinâmica de ambiente
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

def criar_ambiente_estatico():
    AMB = zeros((4,4), dtype=int)


# Criando primeira geração
def cria_geracao1(alfabeto):
    qtd = randrange(2,200)
    G = []
    for q in range(0, qtd):
        tam = randrange(10, 1000)
        Ind = []
        for t in range(0, tam):
            # Ind.append(str ( alfabeto[ randrange( 0, len(alfabeto) ) ] ))
            Ind.append(choice(alfabeto))
        G.append(Ind)
    return G


def mutacao(I, alfabeto):
    I[randrange(0, len(I))] = choice(alfabeto)
    # altera um genes aleatório do individuo, de acordo com o alfabeto
    return I


def crossover(G, alfabeto):
    qi = len(G);
    if (qi%2==1):
        qi += 1
    qi = qi/2
    i = 0
    while i < qi:
        idx = randrange(0, len(G))
        idy = randrange(0, len(G))
        x = G[idx]
        y = G[idy]
        if (len(x) < 3) or (len(y) < 3):
            xy = concatenate([x,y])
            yx = concatenate([y,x])
        else:
            corte1x = randrange(1,len(x))
            corte2x = randrange(corte1x,len(x))
            corte1y = randrange(1, len(y))
            corte2y = randrange(corte1y, len(y))
            xy = concatenate([x[0:corte1x], y[corte1y:corte2y], x[corte2x:len(x)]])
            yx = concatenate([y[0:corte1y], x[corte1x:corte2x], y[corte2y:len(x)]])
        # print 'x= ', x
        # print 'y= ', y
        # print corte1x, corte2x, corte1y, corte2y
        # print 'xy= ', xy,'| yx= ', yx
        # print 'A = ', A
        # print 'B = ', B
        G.append(list(mutacao(xy, alfabeto)))
        G.append(list(mutacao(yx, alfabeto)))
        i += 1
    return G

                  
def fitness(G,wumpus,buracos,ouro,mov_validos):
    pegar = 10
    npegar = -5
    atirar_err = -50
    matar = 1
    voltar = 2
    
    visitados = [(0,0)]
    NG = []
    for cromo in range(0, len(G)):
        In = G[cromo]
        f = 0;
        for gen in range(0, len(In)):
            gene = In[gen]
            # print gene
            """
            cont = 1;
            if len(visitados) > 4:
                for u in range(0, len(visitados)-2):
                    if visitados[len(visitados)-1] == visitados[u]:
                        cont += 1
                        
            f += cont**(-1)
            """
            
             
            if visitados[len(visitados)-1] == ouro:
                f += 500
            if gene == 'N':
                visitados = movimentar(visitados, 1, 0)
                f += avalia_mov(visitados[len(visitados)-1],mov_validos,buracos,wumpus);
            elif gene == 'S':
                visitados = movimentar(visitados, -1, 0)
                f += avalia_mov(visitados[len(visitados)-1],mov_validos,buracos,wumpus);
            elif gene == 'L':
                visitados = movimentar(visitados, 0, 1)
                f += avalia_mov(visitados[len(visitados)-1],mov_validos,buracos,wumpus);
            elif gene == 'O':
                visitados = movimentar(visitados, 0, -1)
                f += avalia_mov(visitados[len(visitados)-1],mov_validos,buracos,wumpus);
            elif gene == 'P': 
                if visitados[len(visitados)-1] in ouro:
                    f += pegar
                else:
                    f += npegar
            elif gene == 'AN':
                pos = visitados[len(visitados)-1]
                mira = [(pos[0] + 1, pos[1])]
                if (mira == wumpus):
                    f += matar
                    wumpus = []
                else:
                    f += atirar_err 
            elif gene == 'AS':
                pos = visitados[len(visitados)-1]
                mira = [(pos[0] - 1, pos[1])]
                if (mira == wumpus):
                    f += matar
                    wumpus = []
                else:
                    f += atirar_err
            elif gene == 'AL':
                pos = visitados[len(visitados)-1]
                mira = [(pos[0], pos[1] + 1)]
                if (mira == wumpus):
                    f += matar
                    wumpus = []
                else:
                    f += atirar_err
            else:
                pos = visitados[len(visitados)-1]
                mira = [(pos[0], pos[1] - 1)]
                if (mira == wumpus):
                    f += matar
                    wumpus = []
                else:
                    f += atirar_err
        # print In            
        # print 'pontuação: ', f
        Inp = []
        Inp.append(In)
        Inp.append(f)
        NG.append(Inp)
    
    # print NG
    
    for i in range(0, len(NG)-1):
        If = NG[i]
        for j in range(i+1, len(NG)):
            if If[1] < NG[j][1]:
                NG[i] = NG[j]
                NG[j] = If
                If = NG[i]
                
    # print NG
    
    G = []
    for k in range(0, (len(NG)/2)):
        G.append(NG[k][0])
        
    return G
    
    # print G
        


def avalia_mov(localizacao,mov_validos,buracos,wumpus):
    cair_buraco = -5
    tem_wumpus = -5
    mover_err = -5
    mover_ok = 10
    f = 0
    if localizacao in mov_validos:
        f = f + mover_ok
    else:
        f = mover_err 
    if localizacao in buracos:
        f = f + cair_buraco
    
    if localizacao in wumpus:
        f = f + tem_wumpus
    
    return f
            
    
def movimentar(visitados,mx,my):
    pos_atual = visitados[len(visitados)-1]
    pos_atual = (pos_atual[0] + mx, pos_atual[1] + my)
    visitados.append(pos_atual)
    return visitados


def solucao(visitados, AMB, ouro, buracos, wumpus):
    seguros = []
    for x in range(0, len(AMB)):
        for y in range(0, len(AMB)):
            if(AMB[x][y] in (0,2,10)):
                seguros.append((x,y))
                
    atual = visitados[len(visitados)-1]
          
    while atual != ouro[0]:
        x = atual[0]
        y = atual[1]
        possibilidades = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        poss_uteis = []
        for k in range(0, len(possibilidades)):
            if possibilidades[k] in seguros:
                poss_uteis.append(possibilidades[k])
                       
        pu_nvisitados = []         
        for k in range(0, len(poss_uteis)):
            if poss_uteis[k] not in visitados:
                pu_nvisitados.append(poss_uteis[k])

        if len(pu_nvisitados) > 0:
            np = choice(pu_nvisitados)
       
        else:
            valor_melhor = -1
            melhor = poss_uteis[0]
            for u in range(0, len(poss_uteis)):
                cont = 0; 
                for f in range(0, len(visitados)):
                    
                    if poss_uteis[u] == visitados[f]:
                        cont += 1
                    
                if cont < valor_melhor or valor_melhor == -1:                 
                    melhor = poss_uteis[u]
                    valor_melhor = cont
                
            np = melhor
            #print (np)
        
        visitados.append(np)
        print(visitados)
        print(AMB)
        
        
            
        atual = visitados[len(visitados)-1]
    #limpa visitados (teste de melhoria)
        
    for u in range(0, len(visitados)-1):
        aux = -1 
        for i in range(u+1, len(visitados)-1):
            print (u, i)
            if (visitados[u]==visitados[i]):
                aux = i;
                print(aux)
        if aux!=-1:
            del visitados[u:aux]
            print (visitados)
    print (visitados)

        
        # print np       
        # print poss_uteis, pu_nvisitados 
        #print seguros      
        #print poss_uteis
       #cl print visitados
    #print(AMB)
        
                
if __name__ == "__main__":
    alfabeto = ('N','N','N','N','S','S','S','S','L','L','L','L','O','O','O','O','P','P','P','P','AS','AN','AL','AO')
    AMB = criar_ambiente()
    AMB[(0,0)] = 10
    # print(AMB)
    mov_validos = []
    visitados = [(0,0)]
    wumpus = []
    buracos = []
    ouro = []
    for x in range(0, len(AMB)):
        for y in range(0, len(AMB)):
            mov_validos.append((x,y))
            if(AMB[x][y] == 3):
                buracos.append((x,y))
            if(AMB[x][y] == 4):
                wumpus.append((x,y))
            if(AMB[x][y] == 2):
                ouro.append((x,y))
    
    # print 'Mov validos: ', mov_validos
    # print 'Buracos: ', buracos
    # print 'Ouro: ', ouro
    # print 'Wumpus: ', wumpus
    
    solucao(visitados, AMB, ouro, buracos, wumpus)
    # print(AMB)
    ### G = cria_geracao1(alfabeto)
    # print G
    # print '--------------------------------------------'
    
    """
    for h in range(0, 100):
        G = fitness(crossover(G, alfabeto),wumpus,buracos,ouro,mov_validos)
    print G
    """
            

        
            
    
            

