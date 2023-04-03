'''
Leia o nome, a matrícula e três notas de um aluno, calcule a média entre 
as notas, sabendo que a primeira nota tem peso 2, a segunda tem peso 3 e 
a terceira tem peso 5. Imprimia de maneira organizada o nome, a matrícula e 
a média obtida pelo aluno.
'''

nome = input("Digite o nome do aluno: ")
mat = input("Digite a matrícula: ")
n1 = float(input("Digite a 1ª nota: "))
n2 = float(input("Digite a 2ª nota: "))
n3 = float(input("Digite a 3ª nota: "))

media = ( (n1*2) + (n2*3) + (n3*5) ) / 10

print("O aluno", nome, " cuja matrícula é ", mat, " obteve média igual a ", media, ".")

# print(f"A média do aluno {nome} de matrícula {mat} é {media}.")

# if media >= 6:
#   print("Aprovado")
# else:
#   print("Reprovado")
