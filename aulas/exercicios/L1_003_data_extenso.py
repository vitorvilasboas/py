"""
Desenvolva um programa com uma função que receba uma data no formato DD/MM/AAAA e devolva uma string com a data por extenso no formato: DD de mesPorExtenso de AAAA. Valide a data e retorne NULL caso a data seja inválida.
"""
def data_por_extenso(data):
    from datetime import datetime as dt
    d = dt.strptime(data, '%d/%m/%Y') # Converte String em Datetime
    mes = 'janeiro' if d.month == 1 else 'fevereiro' if d.month == 2 else 'março' if d.month == 3 else 'abril' if d.month == 4 else 'maio' if d.month == 5 else 'junho' if d.month == 6 else 'julho' if d.month == 7 else 'agosto' if d.month == 8 else 'setembro' if d.month == 9 else 'outubro' if d.month == 10 else 'novembro' if d.month == 11 else 'dezembro' if d.month == 12 else None
    return f'{d.day} de {mes} de {d.year}'

print(f'Data por extenso: {data_por_extenso(input("Informe uma data no formato dd/mm/aaaa: "))}')