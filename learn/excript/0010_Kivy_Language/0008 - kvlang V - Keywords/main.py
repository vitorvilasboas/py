#coding: utf-8

#author: Cláudio Rogério Carvalho Filho

#####################
#### eXcript.com ####
#####################

import kivy
kivy.require("1.9.1")
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout


def funcSelf(x):
    print("funcSelf")
Button.funcSelf = funcSelf


class MinhaTela(BoxLayout):

    def funcRoot(self):
        print("funcRoot")

class Estudo5App(App):

    def funcApp(self):
        print("funcApp")

janela = Estudo5App()
janela.run()