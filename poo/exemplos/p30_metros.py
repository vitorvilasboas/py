"""
Exemplo 008. Solicite do usuário uma medida em metros e mostre-a convertida para centímetros e milímetros (use o padrão format para saída padrão).
"""

met = float(input("Digite uma medida em metros: "))
cent = met * 100
mili = met * 1000

print(f"{met} metros equivale a {cent} centimetros e {mili} milimetros.")

