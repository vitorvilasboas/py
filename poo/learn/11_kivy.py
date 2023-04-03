# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import re
import os
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from view import telas


class Aplicativo(App):
    def build(self):
        sm = ScreenManager()  # instance a new layout manager (gerenciador de layout)

        t1 = telas.Tela1("Vitor1", name='tela1')  # CREATE SCREEN
        t2 = telas.Tela2("Vitor2", name='tela2')
        t3 = telas.Tela3("Vitor3", name='tela3')

        sm.add_widget(t1)  # ADD SCREEN TO SCREEN MANAGER
        sm.add_widget(t2)
        sm.add_widget(t3)

        sm.current = 'tela1'  # define the first current screen
        return sm


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    Builder.load_file("view/screens.kv")
    Aplicativo().run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


