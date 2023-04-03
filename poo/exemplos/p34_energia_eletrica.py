'''
Uma empresa de energia elétrica calcula o valor da conta de luz de acordo com o consumo em Kwh. 
Sabendo que a tarifa cobrada por Kwh de consumo é de R$0.97, crie um programa capaz de ler o CPF 
do consumidor bem como as leituras anterior e atual em kwh. O programa deve calcular o valor a ser 
pago e mostrá-lo junto com o CPF do consumidor e a diferença de consumo atual em Kwh.
'''

tarifa = 0.97
cpf = input("Informe o CPF do consumidor: ")
kw_anterior = float(input("Consumo anterior (Kwh): "))
kw_atual = float(input("Consumo atual (Kwh): "))
consumo = kw_atual - kw_anterior
fatura = consumo * tarifa
print("Para o consumidor de CPF {} foi aferido um consumo de energia elétrica de {:.1f} kwh para o qual o valor a ser pago é de R${:.2f}.".format(cpf, consumo, fatura))