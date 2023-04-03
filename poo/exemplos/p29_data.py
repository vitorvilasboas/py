"""
Exemplo 007. Crie um script Python capaz de ler o dia, o mês e o 
ano de nascimento de uma pessoa e mostrar uma mensagem com a data formatada.
"""

dia = input("Informe o dia do seu nascimento: ")
mes = input("Informe o mês do seu nascimento: ")
ano = input("Informe o ano do seu nascimento: ")

print("Data de nascimento: ", dia, "/", mes, "/", ano)

print("Data de nascimento: {}/{}/{}".format(dia,mes,ano))

print(f"Data de nascimento: {dia}/{mes}/{ano}")