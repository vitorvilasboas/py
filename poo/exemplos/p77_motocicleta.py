"""
Crie uma aplicação que represente uma classe Moto com atributos marca, modelo, cor, marcha_atual, maior_marcha e menor_marcha. Além do método construtor, para permitir a definição de atributos no momento da instanciação do objeto, implemente os métodos:
marchaAcima() - para aumentar 1 marcha de cada vez (somente se a marcha atual for menor que a maior marcha possível).
marchaAbaixo() - para reduzir a 1 marcha por vez (somente se a marcha atual for maior que a menor marcha possível).
imprimir() - imprime os valores de todos os atributos do objeto instanciado de Moto.
Instancie um objeto da classe Moto a partir de valores informados pelo usuário para marca, modelo, cor, marcha_atual, maior_marcha e menor_marcha. Em seguida apresente um menu ao usuário perguntando qual operação ele deseja realizar de 4 possíveis, tais quais:
[1] Reduzir marcha
[2] Aumentar marcha
[3] Imprimir características da moto
[0] Sair
O programa deve executar o método correspondente à opção do menu escolhida pelo usuário e, em seguida, o menu deve ser apresentado novamente até que o usuário indique a opção para Sair do programa.
"""