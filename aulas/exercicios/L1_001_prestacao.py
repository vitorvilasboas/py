"""
Desenvolva um programa que contenha uma função valorPagamento para determinar o valor a ser pago por uma prestação de uma conta. O programa deverá solicitar ao usuário o valor da prestação e o número de dias em atraso e passar estes valores para a função valorPagamento, que calculará o valor a ser pago e devolverá este valor ao programa que a chamou. O programa deverá então exibir o valor a ser pago na tela. Após a execução o programa deverá voltar a pedir outro valor de prestação e assim continuar até que seja informado um valor igual a zero para a prestação. Neste momento o programa deverá ser encerrado, exibindo o relatório do dia, que conterá a quantidade e o valor total de prestações pagas no dia. O cálculo do valor a ser pago é feito da seguinte forma: para pagamentos sem atraso, cobrar o valor da prestação; quando houver atraso, cobrar 3% de multa mais 0,1% de juros por dia de atraso.
"""


def valorPagamento(prestacao, atraso):
    """
    Esta função calcula o valor para pagamento de uma prestação conforme o tempo de atraso
    Parâmetros:
    Entradas: prestacao (em reais), atraso (em dias)
    Saída: Valor da prestação a pagar a partir dos juros e da multa obtidos pelo atraso
    """
    if atraso == 0:
        return prestacao
    else:
        multa = prestacao * 0.03  # 0.03 representa 3% da prestação
        juros = atraso * 0.01 * prestacao  # 0.01 representa 1% da prestacao
        calculo = prestacao + multa + juros
        return calculo


somaPrestacoes = 0
qtdPrestacoes = 0

while True:
    prestacao = float(input("Informe o valor da prestação em reais: R$"))

    if prestacao == 0:
        print("Programa encerrado.")
        break
    else:
        atraso = int(input("Informe a quantidade de dias em atraso: "))
        pagar = valorPagamento(prestacao, atraso)
        print("Valor a pagar: R${:.2f}".format(pagar))
        somaPrestacoes += pagar  # incrementa soma das prestações com prestação atual
        qtdPrestacoes += 1  # mais uma prestação paga

print(f"Nº de prestações processadas: {qtdPrestacoes}")
print("Valor total das prestações: R${:.2f}".format(somaPrestacoes))