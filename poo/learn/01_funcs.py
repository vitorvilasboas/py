"""
Depurar: executar linha a linha para inspeção
Breakpoint define a linha onde a execução será interrompida para inspeção
Teclas de atalho:
    F8      executa linha a linha após primeiro breakpoint (Step over)
    Alt+F8  abre a view Evaluate expression (possibilita avaliar cada variável/objeto)
    F7      próximo passo dentro de um bloco de instrução ou função (Step into)
"""

a = 10
b = 1, 2, 3
x = [a, b]


def function1():
    print("qualquer texto")


def function2():
    function1()


def function3():
    function2()


def function4():
    f4 = 50
    function3()


def function5():
    function4()


function5()

print(a, b, x)

