"""**Exemplo**. Leia os catetos de um triângulo retângulo e calcule sua hipotenusa."""

### equação: h^2 = ca^2 + co^2     # hipotenusa ao quadrado é igual a soma dos quadrados dos catetos
## lembre-se que o python é uma linguagem de tipagem fraca (você não precisa informar o tipo da variável explicitamente)

import math

ca = input("Informe o valor do cateto adjacente: ")
co = input("Informe o valor do cateto oposto: ")

soma_cat = int(co)**2 + int(ca)**2     # soma dos quadrados dos catetos

h = math.sqrt(soma_cat)    # hipotenusa é igual à raiz quadrada da soma dos quadrados dos catetos

print(f"Sendo CA igual a {ca} e CO igual a {co} a hipotenusa é {int(h)}.")