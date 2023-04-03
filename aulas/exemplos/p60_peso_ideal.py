"""
Crie um programa que, tendo como entrada a altura e o sexo de cinco pessoas, calcule e mostre o peso ideal de cada
uma delas, sabendo que o peso ideal para homens é definido por (72,7 ∗ altura) − 58, e para mulheres, por (62,1 ∗ altura) − 44,7.
Ao final, o programa deve mostrar a mulher de menor peso e o homem de maior peso.
"""
# pessoas = ["Vitor", "Caio", "Mércia", "Bruno", "Ygor"]
pessoas = []
for i in range(0, 4):
  h = float(input(f"Informe a altura da {i+1}a pessoa: "))
  s = input(f"Informe o sexo da {i+1}a pessoa [M=masculino ou F=feminino]: ")
  if(s == "M"):
    peso = (72.7 * h) - 58
    print(f"Peso: {peso}")
    p = {'altura':h, 'sexo':s, 'peso':peso}
    pessoas.append(p)
  elif(s == "F"):
    peso = (62.1 * h) - 44.7
    print(f"Peso: {peso}")
    p = {'altura':h, 'sexo':s, 'peso':peso}
    pessoas.append(p)
  else:
    print("Erro ao informar o sexo")

menor = pessoas[0]
maior = pessoas[0]

for p in pessoas:
  if((p['sexo'] == 'M') and (p['peso'] > maior['peso'])):
    maior = p
  if((p['sexo'] == 'F') and (p['peso'] < menor['peso'])):
    menor = p

print(f"Mulher com menor peso: {menor}")
print(f"Homem com maior peso: {maior}")