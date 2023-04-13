"""
Numa competição de salto em distância cada atleta tem direito a cinco saltos e seu desempenho é determinado pela média dos cinco saltos. Desenvolva um programa que leia o nome e as cinco distâncias saltadas por (N) atletas com funções capazes de armazenar os dados dos atletas, calcular e mostrar a média dos saltos de cada atleta. Ao fim da leitura dos dados de um atleta o programa deve perguntar se o usuário deseja inserir dados de outro atleta, e repetir os passos do atleta anterior caso a resposta seja positiva. A entrada de novos dados deve ser encerrada apenas quando a resposta for negativa. Ao final, mostre o nome e a média de saltos do vencedor, ou seja, o atleta com maior distância média nos saltos.
"""
def calcula_medias(atleta: dict):
    """
    Função que calcula a média das distâncias medidas nos 5 saltos realizados por um atleta.
    :param atleta: Dicionário com os dados de um atleta.
    :return: Dicionário com os dados do atleta atualizado com a média
    """
    atleta['media'] = (atleta['salto1'] + atleta['salto2'] + atleta['salto3'] + atleta['salto4'] + atleta[
        'salto5']) / 5
    return atleta

def cadastrar_atleta(atletas):
    # cria um dicionário vazio para armazenar os dados de 1 novo atleta
    atleta = {'nome': input("\nInforme o nome do novo atleta: ").capitalize()}

    # repetição interna para solicitar as 5 distâncias dos saltos do novo atleta
    for cont in range(5):
        # solicita cada salto e salva no dicionário
        atleta[f'salto{cont + 1}'] = float(
            input(f"Informe a distância do {cont + 1}º salto de {atleta['nome']} em metros: "))

    # adiciona o dicionário do novo atleta na lista de atletas
    atletas.append(atleta)
    return atletas


# Função principal da aplicação — chamada automaticamente ao executar
if __name__ == '__main__':
    ## PASSO 1: Cadastrar dados atletas — nome e distâncias dos 5 saltos

    # cria uma lista vazia para armazenar os atletas
    atletas = []

    # repetição (faça enquanto não encontrar o break) — Quantidade de atletas desconhecida!
    while True:
        atletas = cadastrar_atleta(atletas)
        # pergunta se o usuário quer adicionar outro atleta
        escolha = input("\nQuer continuar? s=sim, n=não ")

        # se a resposta não for sim (diferente de y)
        if escolha != 's':

            # PARE! de repetir (encontrou o break)
            break

    ## PASSO 2: Calcular a média de saltos para todos os atletas cadastrados

    print("\n=== Média dos Saltos: === ")

    # Loop percorre a lista de atletas — Aqui a quantidade de atletas já é conhecida!
    for i in range(len(atletas)):
        # chama a função calcularMedias() passando os dados de cada atleta como parâmetro e atualiza a lista com o retorno da função
        atletas[i] = calcula_medias(atletas[i])

        print(f"\t{atletas[i]['nome']}: {atletas[i]['media']} m")

    ## Passo 3: Verificar qual atleta tem a maior média de saltos para eleger o vencedor da competição

    # temporariamente o vencedor passa a ser o 1º atleta cadastrado
    vencedor = atletas[0]

    # print(vencedor)

    # Loop percorre a lista de atletas... ao final vencedor passará a ser o atleta com maior média
    for i in range(len(atletas)):

        # se a média do atleta atual for maior que a média do vencedor até então...
        if atletas[i]['media'] > vencedor['media']:

            # ...vencedor passa a ser o atleta atual
            vencedor = atletas[i]

    # Informa os dados do atleta vencedor
    print(f"\n{vencedor['nome']} é o VENCEDOR por ser o atleta com maior média de salto ({vencedor['media']} metros)!")