'''
Sabendo que em uma escola a Turma C é composta por 50 lista1_alunos e a Turma D por 30 lista1_alunos,
crie um programa Python capaz de ler o percentual de lista1_alunos reprovados na Turma C, o
percentual de aprovados na Turma D, calcular e mostrar:
➔ o número de lista1_alunos reprovados na Turma C;
➔ o número de lista1_alunos reprovados na Turma D;
➔ a percentagem de lista1_alunos reprovados em relação ao total de lista1_alunos das duas turmas.
'''
perc_reprov_c = float(input("Informe o percentual de lista1_alunos reprovados na Turma C: "))
perc_aprov_d = float(input("Informe o percentual de lista1_alunos aprovados na Turma D: "))
qtd_reprov_c = (perc_reprov_c / 100) * 50
qtd_reprov_d = ((100 - perc_aprov_d) / 100) * 30
perc_reprovados = ((qtd_reprov_c + qtd_reprov_d) / 80) * 100
print(f"\nNº lista1_alunos reprovados Turma C: {int(qtd_reprov_c)}")
print(f"Nº lista1_alunos reprovados Turma D: {int(qtd_reprov_d)}")
print("Percentual total de reprovados: {:.1f}%".format(perc_reprovados))
