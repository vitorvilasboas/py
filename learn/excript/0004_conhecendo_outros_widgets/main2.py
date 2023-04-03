#coding: utf-8

#author: Cláudio Rogério Carvalho Filho

#####################
#### eXcript.com ####
#####################

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput

def click():
    # global ed
    print(ed.text)

def build():

    leiaute = FloatLayout()

    ed = TextInput(text="oi")
    global ed
    ed.size_hint = None, None
    ed.height = 300
    ed.width = 400
    ed.top = 550
    ed.right = 500

    bt = Button(text="Clique aqui")
    bt.on_press = click
    bt.size_hint = None, None
    bt.height = 50
    bt.width = 200
    bt.top = 200
    bt.x = 200
    bt.y = 100

    leiaute.add_widget(bt)
    leiaute.add_widget(ed)

    return leiaute

janela = App()
janela.build = build

from kivy.core.window import Window
Window.size = (600, 600)

janela.run()