"""
Crie um programa que simule um jogo ‘pedra, papel, tesoura’ entre o usuário e o computador.
O programa deve encerrar somente quando o usuário desejar.
"""
import random
import time
opcoes = ['pedra', 'papel', 'tesoura']
cont_user, cont_comput, jogadas = 0, 0, 0
while True:
    print('''Escolha uma das opções: 
          1 - Pedra
          2 - Papel
          3 - Tesoura
          0 - Sair 
          ''')
    usuario = int(input("Opção: "))
    comput = random.choice(opcoes)
    # comput = random.randint(1, 3)
    if usuario != 0:
        if usuario not in [1, 2, 3]:
            print("Opção inválida!")
        else:
            jogadas += 1
            if (usuario == 1 and comput == 'tesoura') or (usuario == 2 and comput == 'pedra') or (
                    usuario == 3 and comput == 'papel'):
                print(f"Computador escolheu {comput}, logo você venceu!\n")
                cont_user += 1
            elif (usuario == 1 and comput == 'papel') or (usuario == 2 and comput == 'tesoura') or (
                    usuario == 3 and comput == 'pedra'):
                print(f"Computador escolheu {comput} e venceu!\n")
                cont_comput += 1
            else:
                print(f"Computador escolheu {comput} e, portanto, houve empate!\n")
    else:
        print("Saindo", end=" ")
        for i in range(0, 3):
            time.sleep(1.5)
            print(".", end=" ")
        break
print(f"\nJogadas:: {jogadas}")
print(f"Placar:: computador = {cont_comput}  || você = {cont_user}")
