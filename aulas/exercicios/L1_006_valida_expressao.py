"""
Desenvolva um programa que leia uma expressão matemática fornecida pelo usuário e verifique se a abertura e o fechamento de parênteses está correto na ordem e na quantidade.
"""

exp = input("Defina uma expressão matemática: ")
abre = 0
fecha = 0
for termo in exp.split('='): # termo = exp.split('=')[0] e exp.split('=')[0]
    valida = False
    for c in termo:
        if c == '(':
            abre += 1
            valida = True
        if c == ')':
            fecha += 1
            if not valida:
                print("Expressão inválida... um ')' precede qualquer '('")
                break
            else: valida = False
    if abre == fecha: valida = True
    else:
        print("Expressão inválida... Quantidade de '(' diferente de ')'.")
        break

if valida: print("EXPRESSÃO VÁLIDA!!!")


# try:
#     resultado = eval(expressao)
#     print(f"A expressão {expressao} é válida e o resultado é {resultado}.")
# except (SyntaxError, TypeError):
#     print(f"A expressão {expressao} é inválida.")
