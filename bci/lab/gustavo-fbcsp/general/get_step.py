# -*- coding: utf-8 -*-
def get_step(n_bins, n_bands, overlap):
	# Função usada para obter o step (intervalo de coeficientes entre duas bandas)

	# Estimativa inicial do step, que seria o número de bins sobre o número de bandas truncado em múltiplos de 2
	step = (n_bins / n_bands / 2) * 2
	size = step * overlap

	# Porém as vezes o resultado retornado faz com que a última banda tenha bins não existentes.
	# Para corrigir isso usei um loop que verifica o último bin da última banda e se ele não for menor ou igual
	# ao número de bins o step é reduzidor por 2. Provavelmente existe alguma maneira de achar isso de forma
	# não-iterativa mas como o tempo em que isso é feito é desprezível não procurei otimizar essa parte
	while True:
		last_end = (n_bands-1) * step + size
		if last_end <= n_bins:
			break
		step -= 2
		size = step * overlap  # add by Vitor, cleison.py based

	return step
