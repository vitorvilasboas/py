# Exemplicação de I/O em python

nome = input("Qual o seu nome: ") # atribuição de valor à variável nome
idade = input("Qual a sua idade: ") # todo valor de entrada via função input é do tipo string (cadeia)
altura = input("Qual a sua altura: ")
peso = input("Qual o seu peso: ")

print("Olá", nome, ", seja bem vindo. \nSei que você tem", idade, "anos, \npossui", altura, "metros de altura \ne seu peso é ", peso, "kilos.")

# Conversão str para int
calculo = int(idade) - 2  # a função int() converte o argumento p/ o tipo inteiro

print("Resultado idade: ", calculo)

# Conversão str para float
calculo2 = float(altura) - 1 # a função float() converte o argumento p/ o tipo real

print("Resultado altura: ", calculo2)