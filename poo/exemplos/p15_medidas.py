'''
Solicite do usuário uma medida em metros e mostre-a convertida para centímetros e milímetros.
'''

met = float(input("Digite uma medida em metros: "))
cent = met * 100
mili = met * 1000

print(met, "metros equivale a", cent, "centimetros e", mili, "milimetros.")

# print(f"{met} metros equivale a {cent} centimetros e {mili} milimetros.")