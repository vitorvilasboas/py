from kivy.app import App
from kivy.uix.widget import Widget

from kivy.interactive import InteractiveLauncher

from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')
Config.set('modules', 'console', '')

class EstudoApp(App):
    def build(self):
        return Widget()

j = EstudoApp()
i = InteractiveLauncher( j )

i.run()

#from kivy.uix.button import Button
#bt = Button(text="estudo")

#i.root.add_widget(bt)

#bt.text = "oi"