#coding: utf-8

#author: Cláudio Rogério Carvalho Filho

#####################
#### eXcript.com ####
#####################

from kivy.app import App
from kivy.uix.textinput import TextInput

def build():
    return TextInput(text="Componente TextInput")

janela = App()
janela.build = build
janela.run()
