#coding: utf-8

#author: Cláudio Rogério Carvalho Filho

#####################
#### eXcript.com ####
#####################

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

def click():
    print(ed.text)

def build():

    layout = FloatLayout()

    ed = TextInput(text="eXcript")
    global ed
    ed.size_hint = None, None
    ed.height = 300
    ed.width = 400
    ed.x = 60
    ed.y = 250

    bt = Button(text="Clique aqui")
    bt.size_hint = None, None
    bt.width = 200
    bt.height = 50
    bt.y = 150
    bt.x = 170

    bt.on_press = click

    layout.add_widget(ed)
    layout.add_widget(bt)

    return layout


janela = App()
janela.title = "eXcript"

from kivy.core.window import Window
Window.size = 600,600

janela.build = build
janela.run()

