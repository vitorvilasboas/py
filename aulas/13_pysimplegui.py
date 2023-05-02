import PySimpleGUI as sg

layout = [
    [sg.Text("Informar: ")], # primeira linha da tela
    [sg.InputText(key="campo_informar")], # segunda linha da tela
    [sg.Button("Enviar"), sg.Button("Cancelar")], # terceira linha da tela
    [sg.Text("", key="campo_saida")]
]

janela = sg.Window("Título da janela", layout)

while True:
    evento, valores = janela.read() # lê constantemente tudo que acontece dentro da janela
    if evento == sg.WIN_CLOSED or evento == "Cancelar":
        break ## se clicou no X
    if evento == "Enviar":
        pegar = valores["campo_informar"]
        janela["campo_saida"].update(f"O texto é: {pegar}")

janela.close()