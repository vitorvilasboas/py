# -*- coding: utf-8 -*-

#-%% Operadores relacionais
1 > 2
1 < 2
1 >= 1
1 <= 4
1 == 1
1 != 1
'ppca' == 'PPCA'


#-%% Operadores lógicos
teste = not (True)
teste1 = (1 > 2) and (2 < 3)
teste2 = (1 > 2) or (2 < 3)
teste3 = (1 == 2) or (2 == 3) or (4 == 4)

if teste:
    print('Bloco True')
else:
    print('Bloco False')
    
# ternário
print('Bloco True' if teste else 'Bloco False')

ternario = ('Ok' if teste else 'Ops')


lista = ['A','B','C']

if lista[0] == 'A':
    print('É a')
elif lista[0] == 'b':
    print('É b')
else:
    print('Nem a nem b')
    

# if,elif, else
if 1 < 2:
    print('Sim!')
if 1 < 2:
    print('Primeiro')
else:
    print('Último')

if 1 > 2:
    print('Primeiro')
else:
    print('Último')

if 1 == 2:
    print('Primeiro')
elif 3 == 3:
    print('Meio')
else:
    print('Último')
    


if teste2:
    pass

''