#coding: utf-8

#author: Cláudio Rogério Carvalho Filho

#####################
#### eXcript.com ####
#####################

import kivy
kivy.require("1.9.1")
from kivy.app import App

import Interactive

code = """

BoxLayout:
    Button:
        text: "1"
    Button:
        text: "2"

"""

from kivy.lang import Builder

Builder.load_file()

class Estudo6App(App):
    def build(self):
        return Builder.load_string(code)

janela = Estudo6App()
janela.run()