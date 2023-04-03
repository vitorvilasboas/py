"""
Exemplo: Sistema de Recomendação de Cursos. Projete e implemente um sistema de recimendação, a partir de uma lista de cursos, cada qual com título, duração em horas, categoria e gratuidade. Após estabelecer uma lista de dicionários de cursos, o sistema deve consultar as preferências do usuário como: categoria de interesse, tempo mínimo de duração e Gratuidade, construir seu perfil e recomendar os cursos, no dicionário, que atendam a esse perfil. Abaixo, um exemplo de saída do sistema:

Olá, Fulano, com base no seu perfil, achamos que você iria gostar dos seguintes cursos:
PROGRAMAÇÃO:
- Programação Python (90 horas)
- Desenvolvimento de Plugins para WordPress (160 horas)
IA:
- Machine Learning com Python (100 horas)
"""

nome = input("Como gostaria de ser chamado(a): ")

cursos = [{'titulo': 'Programação Python', 'duracao': 90, 'categoria': 'Programação', 'gratuidade': True},
          {'titulo': 'Machine Learning com Python', 'duracao': 100, 'categoria': 'IA', 'gratuidade': False},
          {'titulo': 'Desenvolvedor Web Full Stack', 'duracao': 160, 'categoria': 'Programação', 'gratuidade': False},
          {'titulo': 'Introdução ao Google Ads', 'duracao': 25, 'categoria': 'Marketing digital', 'gratuidade': True},
          {'titulo': 'E-commerce e Vendas Online', 'duracao': 40, 'categoria': 'Vendas', 'gratuidade': False},
          {'titulo': 'Canva para Empreendedores', 'duracao': 60, 'categoria': 'Marketing digital', 'gratuidade': True},
          {'titulo': 'Desenvolvimento de Plugins para WordPress', 'duracao': 70, 'categoria': 'Programação', 'gratuidade': False}]

lista_categorias = list(set([ cursos[i]['categoria'] for i, curso in enumerate(cursos) ]))

print('\nCategorias de Cursos: ')
for i,cat in enumerate(lista_categorias): print(f"{i}. {cat}")

id_categorias = [ int(n.strip()) for n in input("\nInforme o ID das categorias de seu interesse (separe com vírgulas): ").split(',') ]
tempo = float(input("Quanto tempo gostaria de estudar (duração mínima em horas)? "))
# gratuidade = input("Preferência por Cursos Gratuitos [S/N]? ")
pagos = False if input("Incluir Cursos Pagos [S/N]? ").upper() == "N" else True

categorias_selecionadas = [ lista_categorias[id] for id in id_categorias ]

print(f"\nNome: {nome}")
print(f"Categorias:", end="")
for cat in categorias_selecionadas: print(f" {cat},", end="")
print(f"\nDuração mínima: {tempo} horas")
print(f"Incluir Pagos: {'SIM' if pagos else 'NÃO'}")

print(f"\n{nome}, com base no seu perfil, achamos que você iria gostar dos seguintes cursos: ")

for cat in categorias_selecionadas:
  print(f"\n{cat}:")
  for curso in cursos:
    if (curso['gratuidade'] or pagos) and (cat.capitalize() == curso['categoria'] and tempo <= curso['duracao']):
      print(f"– {curso['titulo']} ({curso['duracao']} horas)")