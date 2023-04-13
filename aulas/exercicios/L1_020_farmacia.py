"""
Os salários dos vendedores de uma farmácia são calculados com base em um valor fixo de R$1200,00 mais uma comissão de 9% do valor bruto de suas vendas no mês. Desenvolva um programa capaz de ler o nome, o cpf e o valor bruto de vendas no mês de cada um dos 8 vendedores contratados pela farmácia, calcule e armazene junto a esses dados os respectivos salários. Ao final, informe:
    a. O valor total de vendas brutas no mês em R$
    b. A média em R$ das vendas registradas pelos vendedores no mês
    c. O valor total de salários pagos pela farmácia no mês em R$
    d. A média de salários do mês em R$
    e. Os dados do vendedor com maior salário no mês
    f. Os dados do vendedor com menor salário no mês
    g. Os dados dos vendedores cujo salário foi superior à R$3000,00 em ordem decrescente de salários.
"""
SALARIO = 1200
lista = []
for i in range(8):
    f = {'nome': input(f"Informe o nome do {i+1}º vendedor: ")}
    f['cpf'] = input("CPF: ")
    f['vendas'] = float(input("Valor bruto vendas mensal: R$"))
    f['salario'] = SALARIO + (f['vendas'] * 0.09)

print("::: RELATÓRIO :::")
print("Total vendas no mês: R${:.2f}".format(sum([x['vendas'] for x in lista])))
print("Média vendas no mês: R${:.2f}".format(sum([x['vendas'] for x in lista])/len(lista)))
print("Total salários no mês: R${:.2f}".format(sum([x['salario'] for x in lista])))
print("Média salários no mês: R${:.2f}".format(sum([x['salario'] for x in lista])/len(lista)))

maior_sal = sorted(lista, key=lambda x:x['salario'])[-1]
menor_sal = sorted(lista, key=lambda x:x['salario'])[-1]
print("Maior bônus: R${:.2f} [{} - CPF:{}]".format(maior_sal['salario'], maior_sal['nome'], maior_sal['cpf']))
print("Menor bônus: R${:.2f} [{} - CPF:{}]".format(menor_sal['salario'], menor_sal['nome'], menor_sal['cpf']))

print("Vendedores c/ salário > R$3000:")
for f in [x for x in lista if x['salario'] > 3000]:
    print("{} - CPF:{} - Salário: R${:.2f}".format(f['nome'], f['cpf'], f['salario']))

