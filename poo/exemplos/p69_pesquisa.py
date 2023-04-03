"""
Exemplo: Desenvolva um programa que simule uma pesquisa entre os habitantes de uma região solicitando o nome, a idade, o sexo e o salário de uma quantidade, à princípio, indeterminada de entrevistados. A entrada de dados deve ser encerrada quando for digitado “fim” para o campo nome ou uma idade negativa. Ao final, o programa deve apresentar um relatório com:
a média de salário do grupo;
quantos homens e quantas mulheres há no grupo
os dados das pessoas com maior e menor idade do grupo;
a quantidade de mulheres com salário até R$100,00.
quantos homens possuem mais de 20 anos.

"""

def menu():
    print("""
        === Senso IBGE 2022 ===
        Escolha uma das opções: 
        (1) Registrar entrevista
        (2) Mostrar média de salário
        (3) Mostrar quantidade de homens e mulheres
        (4) Listar maior e menor idades
        (5) Listar mulheres com salário até R$100,00
        (6) Listar homens com idade maior que 20 anos
        (7) Sair
    """)
    opcao = int(input("> "))
    return opcao

def cadastrarEntrevistra():
    pass

def mostrarMediaSalarios():
    pass

def mostrarQtdPorSexo():
    pass

def listarMaiorMenorIdades():
    pass

def listarMulheresSalario100():
    pass

def listarHomensIdade20():
    pass

if __name__ == '__main__':
    while True:
        opcao = menu()
        if opcao == 1:
            cadastrarEntrevistra()
        elif opcao == 2:
            mostrarMediaSalarios()
        elif opcao == 3:
            mostrarQtdPorSexo()
        elif opcao == 4:
            listarMaiorMenorIdades()
        elif opcao == 5:
            listarMulheresSalario100()
        elif opcao == 6:
            listarHomensIdade20()
        elif opcao == 7:
            break
        else:
            print("Opção inválida!!!")