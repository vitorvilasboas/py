from random import choice

cave_numbers = range(1, 21)
wumpus_location = choice(cave_numbers)
player_location = choice(cave_numbers)
while player_location == wumpus_location:
    player_location = choice(cave_numbers)
print("Bem vindo ao Mundo de Wumpus!")
print("Você pode ver ", len(cave_numbers), " quadros")
while True:
    print("Você está no quadro ", player_location)
    if(player_location == wumpus_location - 1) or (player_location == wumpus_location + 1):





