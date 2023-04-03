# Exercícios

# Quanto é 7 elevado na potência 4?
7**4

# Quebre a seguinte string em uma lista: 
s = "Olá, Pai!"
s.split()

# Dada as variáveis, Use .format() para printar a seguinte frase: O diâmetro da terra é de 12742 kilômetros.
planeta = "Terra"
diametro = 12742
print("O diâmetro da {} é {} kilômetros.".format(planeta,diametro))

# Dada a lista abaixo, use indexação para obter apenas a string "ola". 
lst = [1,2,[3,4],[5,[100,200,['olá']],23,11],1,7]
lst[3][1][2][0]

# Dado o seguinte dicionário aninhado, extraia a palavra "hello"
d = {'k1':[1,2,3,{'café':['banana','mulher','colher',{'alvo':[1,2,3,'olá']}]}]}
lst[3][1][2][0]

# Qual a principal diferença entre um dicionário e uma tupla?

# Construa uma função que retire o domínio dado um e-mail no seguinte formato: user@domain.com  
# Por exemplo, passando como parâmetro "user@domain.com" retornaria: domain.com
def obterDominio(email):
    return email.split('@')[-1]
obterDominio('user@domain.com')

# Crie uma função básica que retorna True se a palavra 'dog' estiver contida na string de entrada. 
# Não se preocupe com os casos de extremos, como uma pontuação que está sendo anexada à palavra cão, mas que seja sensível à caixa. 
def encontreCachorro(st):
    return 'cachorro' in st.lower().split()
encontreCachorro('Há um cachorro aqui?')

# Crie uma função que conta o número de vezes que a palavra "dog" ocorre em uma string. Novamente ignore os casos extremos.
def contaCachorro(st):
    count = 0
    for word in st.lower().split():
        if word == 'cachorro':
            count += 1
    return count
contaCachorro('Esse cachorro é mais rápido que o outro cachorro.')

# Use expressões lambda e a função filter () para filtrar as palavras de uma lista que não começa com a letra 's'. 
# Por exemplo: seq = ['sopa','cachorro','salada','gato','ótimo']
# Deveria ser filtrado para: ['sopa','salada']
seq = ['sopa','cachorro','salada','gato','ótimo']
list(filter(lambda word: word[0]=='s',seq))

#%% Problema final
"""
Você está dirigindo um pouco rápido demais, e um policial para você. 
Escreva uma função para retornar um dos 3 resultados possíveis: "Sem multa", "Pequena multa" ou "Multa Grande".
Se a sua velocidade for igual ou inferior a 60, o resultado é "Sem multa". Se a velocidade for entre 61 e 80 inclusive, 
o resultado é "Multa Pequena". Se a velocidade é de 81 ou mais, o resultado é "Multa Grande". 
A menos que seja seu aniversário (codificado como um valor booleano nos parâmetros da função) 
- em seu aniversário, sua velocidade pode ser 5 maior em todos os casos.
"""
def capturar_velocidade(velocidade, aniversario):
    if aniversario:
        velocidade = velocidade - 5
    else:
        velocidade = velocidade
    
    if velocidade > 80:
        return 'Multa Pequena'
    elif velocidade > 60:
        return 'Multa Grande'
    else:
        return 'Sem Multa'
capturar_velocidade(65,False)
capturar_velocidade(81,False)