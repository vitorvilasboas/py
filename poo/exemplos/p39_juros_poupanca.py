'''
Desenvolva um programa que receba o valor de um depósito em poupança, calcule e mostre o
valor após cinco meses de aplicação na poupança, sabendo que a poupança rende 8.5% ao mês.
'''
deposito = float(input("Informe o valor depositado na poupança: R$"))
rendimento = deposito * 0.085 * 5
print("Valor depositado mais rendimentos após 5 meses: R${:.2f}".format(deposito + rendimento))