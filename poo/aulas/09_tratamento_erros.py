# escopo de variáveis
def func(b):
    # global n # utiliza a variável global
    n = b ** 2
    return n

n = 2

x = func(n)

print(n, x)

try:
    xx = func(n)
except:
    print('Falha')
else:
    print(xx)
finally:
    print('Exibe sempre')