"""
Pergunte ao usuário seu nome completo e
informe o tempo que o usuário levou para responder.
"""

## Python é case sensitive
import time
inicio = time.time()
nome = input("Qual seu nome completo?")
final = time.time()
duracao = final - inicio
print(f"Ok {nome} você levou {round(duracao,1)} segundos para responder.")