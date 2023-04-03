#coding: utf-8

#author: Cláudio Rogério Carvalho Filho

#####################
#### eXcript.com ####
#####################

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout


def click():
    print("O botão foi clicado")

def build():

    leiaute = FloatLayout()

    bt = Button(text="Clique aqui")
    bt.on_press = click
    bt.size_hint = None, None
    bt.height = 50
    bt.width = 100
    bt.top = 200
    # bt.x = 300
    bt.right = 300


    leiaute.add_widget(bt)

    return leiaute

janela = App()
janela.build = build
janela.run()