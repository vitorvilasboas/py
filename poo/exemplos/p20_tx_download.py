"""
Crie um script Python capaz de ler o tamanho de um arquivo para download (em MB) 
e a velocidade de um link de Internet (em Mbps), calcular e informar o tempo 
aproximado de download do arquivo usando este link (em minutos).
"""
tamanho = float(input("Tamanho do arquivo (MB): "))

velocidade = float(input("Velocidade de download (Mbps): "))

t_segundos = tamanho / velocidade

t_minutos = t_segundos / 60   # converte o tempo em segundos para minutos 

qtd_minutos = int(t_segundos // 60)  # captura qtos minutos há no tempo

# qtd_segundos = int((t_minutos - qtd_minutos) * 60)  # captura qtos segundos restam no tempo

qtd_segundos = t_segundos % 60  # idem usando o operador resto da divisão

print("Tempo estimado para download:", str(qtd_minutos) + ":" + str(qtd_segundos))