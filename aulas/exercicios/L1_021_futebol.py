"""
Desenvolva um programa para gerenciar as estatísticas de gols num campeonato de futebol. O programa deve ler o nome do jogador, o time ao qual ele pertence, quantas partidas ele disputou e a quantidade de gols feitos durante o campeonato, armazenando os dados numa única coleção de dados. Ao final, mostre:
    a. A quantidade de jogadores cadastrados
    b. O total de gols feitos durante o campeonato.
    c. Os dados do artilheiro do campeonato.
    d. Os dados do jogador com mais partidas disputadas.
    e. Os dados do jogador com maior média de gols por partidas disputadas.
"""
jogadores = []
while True:
    if input("Registrar novo jogador? [S]im ou [N]ão: ").upper() != 'S': break
    j = {'nome': input(f"Informe o nome do {i+1}º jogador: "),
        'time': input("Time: "),
        'partidas': int(input("Nº partidas disputadas: ")),
        'gols': int(input("Nº gols disputadas: "))}

print(":: ESTATÍSTICAS ::")
print(f"Qtd jogadores: {len(jogadores)}")
print(f"Total Gols: {sum([j['gols'] for j in jogadores])}")

x = sorted(jogadores, key=lambda j: j['gols'])[-1]
print(f"Artilheiro: {x['nome']} ({x['time']}) - {x['gols']} gols")

x = sorted(jogadores, key=lambda j: j['partidas'])[-1]
print(f"Jogador Mais Partidas: {x['nome']} ({x['time']}) - {x['partidas']} partidas")

for i in range(len(jogadores)): jogadores[i]['media'] = jogadores[i]['gols']/jogadores[i]['partida']
x = sorted(jogadores, key=lambda j: j['media'])[-1]
print(f"Jogador Maior Média Gols: {max([j['gols']/j['partidas'] for j in jogadores])} gols/partida")