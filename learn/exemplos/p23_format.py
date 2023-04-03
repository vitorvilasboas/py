# -*- coding: utf-8 -*-
a, b, c = 1, 2, 7.    # atribuição conjunta
print(a)

a += 2     # incremento de 2 no valor de a  (o mesmo que a = a + 2)
print(a)

a -= 3     # decremento de 3 no valor de a  (o mesmo que a = a - 3)
print(a)

# Quebra de linha (\n)

# Por padrão a quebra de linha determina o término de uma instrução
print("oi")
print("tchau")

# o operador \n define uma quebra de linha
print("oi\ntchau");

print("Meu nome é \n Vitor")

# múltiplas instruções em uma mesma linha - devem ser separadas por ";"
print("oi\ntchau"); x = 5; print(x)

# Formatações alternativas para saída padrão com a função print()

nome = 'Vitor'
sobrenome = 'Vilas Boas'
idade = 34

# função de saída padrão com concatenação por vírgula (,)
print("Meu nome é ", nome, " e meu sobrenome é ", sobrenome, ". Tenho ", idade, " anos de idade.")   

# função de saída com método FORMAT
print("O nome informado é {} e a idade informada é {} anos!".format(nome, idade)) 

# função de saída com método FORMAT (ref por variáveis auxiliares)
print("O nome informado é {n}, e a idade informada é {i} anos!".format(n=nome, i=idade))  

# função de saída com método FORMAT (ref por índices)
print("O nome informado é {0}, e a idade informada é {1} anos!".format(nome, idade))  

# função de saída com método FORMAT abreviado: interpolação fstrings (pós versão 3.6)
print(f"O nome informado é {nome} e a idade informada é {idade} anos!")



# função de saída baseada no endereço de memória 
# %s=string  %d=inteiro/decimal  %f=float
print("O nome informado é %s e a idade informada é %d anos!" %(nome, idade)) 

x, y = 4, 3
print('A soma é %d, o produto é %d e a divisão é %.2f' % (x+y, x*y, x/y)) 
# .2f = duas casas após a vírgula

print('A soma é {}, o produto é {} e a divisão é {:.2f}'.format(x+y, x*y, x/y), end='\n\n\n')  
# 'end' define o que irá ocorrer após a exibição (neste caso 3 quebras de linha)

print('='*20)  # operadores aritméticos na função de saída padrão

# saída FORMAT espaçada - 20 espaços após o argumento
print("Meu nome é {:<20}".format(nome))

# saída FORMAT espaçada - 20 espaços antes do argumento
print("Meu nome é {:>20}".format(nome))   

# saída FORMAT espaçada - argumento centralizado em 20 espaços (10 espaços antes e 10 depois)
print("Meu nome é {:^20}".format(nome))   