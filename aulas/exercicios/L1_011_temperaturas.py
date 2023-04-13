"""
Desenvolva um programa que leia as temperaturas médias dos 12 meses do ano armazenando-as em uma lista. Em seguida, calcule a média anual das temperaturas e mostre os meses cujas temperaturas médias foram superiores à média anual na ordem crescente por temperatura.
"""
import numpy as np
meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
temperaturas = [ float(input(f"Informe a temperatura média do mês de {m} em ºC: ")) for m in meses ]
media = np.mean(temperaturas)
print("A média de temperatura no ano foi de {:.2f}ºC.".format(media))
print("Meses que registraram temperatura maior que a média: ", end='')
for m,t in zip(meses, temperaturas):
    if t > media: print(m, end=' ')