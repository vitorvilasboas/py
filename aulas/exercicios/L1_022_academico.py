"""
Desenvolva um programa que simule um sistema acadêmico cuja interface inicial corresponda a um menu com as seguintes opções:
    1 - Cadastrar aluno
    2 - Buscar aluno
    3 - Emitir relatório completo
    0 - Sair
Na opção 0, o programa deverá ser encerrado após a confirmação do usuário do tipo “Tem certeza que deseja sair? S/N”; Na opção 1, o programa deve ler e armazenar em uma coleção de dados o nome, a matrícula, a série, a turma e as quatro notas bimestrais de um aluno, além de calcular e armazenar a média ponderada das notas de cada aluno de acordo com a fórmula: MP = (nota1*3 + nota2*2 + nota3*3 + nota4*2)/10. Ainda na opção 1 e a partir da média calculada, o programa deve verificar e armazenar o conceito e a situação do aluno conforme tabela abaixo:
    Nota        Conceito    Situação
    0,0 a 3,99  D           Reprovado
    4,0 a 5,99  C           Recuperação
    6,0 a 7,99  B           Aprovado
    8,0 a 10,0  A           Aprovado com mérito
Na opção 2, o programa deve permitir que o usuário busque por um aluno através de sua matrícula e mostrar todos os dados desse aluno caso o encontre na coleção de dados; Na opção 3, o programa deve mostrar um relatório completo que inclui mostrando:
    a. A quantidade e a porcentagem de alunos classificados em cada conceito;
    b. A quantidade e a porcentagem de alunos não aprovados;
    c. A média de notas da turma (média das MPs de todos os alunos);
    d. A média de notas dos alunos não aprovados;
    e. Os dados do aluno com maior MP;
    f. Os dados do aluno com menor MP.
"""