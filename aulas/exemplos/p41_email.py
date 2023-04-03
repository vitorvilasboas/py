"""
Construa uma função que retorne o domínio de um e-mail fornecido pelo usuário. Por exemplo, `para um email
'user@domain.com' passado como parâmetro a função retornaria 'domain.com'.
"""

def obterDominio(email):
    return email.split('@')[-1]


email = input("Informe seu e-mail: ")
obterDominio('user@domain.com')