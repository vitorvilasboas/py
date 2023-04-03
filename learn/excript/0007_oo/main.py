#coding: utf-8

#author: Cláudio Rogério Carvalho Filho

#####################
#### eXcript.com ####
#####################

from kivy.app import App
from kivy.metrics import sp
from kivy.uix.label import Label


class MeuProgra(App):

    def build(self):

        sp()

        return Label()

MeuProgra().run()

