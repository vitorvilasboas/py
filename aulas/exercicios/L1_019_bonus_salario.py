"""
Uma empresa resolveu bonificar seus colaboradores em reconhecimento ao bom resultado alcançado no ano anterior e precisa de uma aplicação capaz projetar o gasto total com o pagamento destes bônus. Após reuniões chegou-se às seguintes bases de cálculo:
    I. Cada funcionário receberá o equivalente a 20% do seu salário bruto de dezembro;
    II. O piso do bônus será de R$100,00, isto é, aqueles funcionários cujo salário for muito baixo, recebem um reajuste mínimo neste valor.
Neste contexto, desenvolva um programa que leia o nome e o salário de N colaboradores, calcule e armazene o valor do bônus colaborador de acordo com as bases definidas. A leitura deve ser encerrada quando o usuário fornecer um nome vazio (“ “). Ao final, apresente:
    a. Nome, salário e bônus concedido a cada funcionário;
    b. A quantidade de funcionários registrados;
    c. O valor total a ser gasto com bonificação considerando todos os funcionários;
    d. A porcentagem de funcionários que receberão o valor mínimo de bônus (R$100,00);
    e. O maior valor pago como bônus.
"""

lista = []
while True:
    nome = input("Informe o nome do funcionário: ")
    if nome == '': break
    salario = float(input(f"Informe o salário do {nome}: R$"))
    bonus = max([100, salario*0.2])
    f = {'nome': nome, 'salario': salario, 'bonus': bonus}
    lista.append(f)

print("::: Funcionários :::")
for f in lista: print("{} - Salário: R${:.2f} - Bônus: R${:.2f}".format(f['nome'], f['salario'], f['bonus']))
print(f"Total de funcionários: {len(lista)}")
print("Soma dos bônus: R${:.2f}".format(sum([f['bonus'] for f in lista])))
print(f"Porcentagem bônus mínimo: R${len(['' for f in lista if f['bonus'] == 100])/len(lista)*100}")
print("Maior bônus: R${:.2f}".format(max([f['bonus'] for f in lista])))