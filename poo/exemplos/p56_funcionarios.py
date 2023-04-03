"""
Os funcionários de uma vitor_poo receberam um bônus no salário conforme o cargo.
    Cargo           Código    Bônus
    Técnico         1         R$500,00
    Gerente         2         R$300,00
    Administrativo  3         R$200,00
Crie um programa que leia o código do cargo e o salário atual de 10 funcionários,
calcule e imprima o salário acrescentado do bônus recebido por cada um deles.
"""
soma_salarios = 0
for i in range(0, 5):
  cargo = int(input("""
                Informe o código do cargo do funcionário
                1 - Técnico
                2 - Gerente
                3 - Administrativo
                : """))
  if(cargo == 1):
    salario = float(input("Informe o salário do funcionário: R$"))
    bonus = 500
    salario = salario + bonus
    print(f"O salário bonificado é R${salario}")
    soma_salarios = soma_salarios + salario
  elif(cargo == 2):
    salario = float(input("Informe o salário do funcionário: R$"))
    bonus = 300
    salario = salario + bonus
    print(f"O salário bonificado é R${salario}")
    soma_salarios = soma_salarios + salario
  elif(cargo == 3):
    salario = float(input("Informe o salário do funcionário: R$"))
    bonus = 200
    salario = salario + bonus
    print(f"O salário bonificado é R${salario}")
    soma_salarios = soma_salarios + salario
  else:
    print("Código inválido!!")

print("A soma de todos os salários bonificados é R${:.2f}".format(soma_salarios))
