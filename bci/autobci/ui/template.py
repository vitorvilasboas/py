# coding: utf-8
from kivy.factory import Factory
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.properties import StringProperty, NumericProperty, OptionProperty, BooleanProperty
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.behaviors.button import ButtonBehavior

__all__ = ('StdSettingsContainer', 'StdSettingItem',
           'StdSettingBoolean', 'StdSettingSlider', 'StdSettingStringLong',
           'StdSettingString', 'StdSettingTitle','StdSettingSpinner',
           'MyButton', 'MySmoothButton', 'MyStrokeButton')

class MyButton(ButtonBehavior,Label):
    pass

class MyStrokeButton(Button):
    pass

class MySmoothButton(Button):
    pass

class StdSettingsContainer(GridLayout):
    pass

class StdSettingItem(GridLayout):
    title = StringProperty('<No title set>')
    desc = StringProperty('')

class StdSettingTitle(Label):
    title = StringProperty('<No title set>')
    desc = StringProperty('')

class StdSettingBoolean(StdSettingItem):
    button_text = StringProperty('')
    value = BooleanProperty(False)

class StdSettingString(StdSettingItem):
    value = StringProperty('')

class StdSettingStringLong(StdSettingItem):
    value = StringProperty('')

class EditSettingPopup(Popup):
    def __init__(self, **kwargs):
        super(EditSettingPopup, self).__init__(**kwargs)
        self.register_event_type('on_validate')

    def on_validate(self, *l):
        pass

class StdSettingSlider(StdSettingItem):
    min = NumericProperty(0)
    max = NumericProperty(100)
    type = OptionProperty('int', options=['float', 'int'])
    value = NumericProperty(0)

    def __init__(self, **kwargs):
        super(StdSettingSlider, self).__init__(**kwargs)
        self._popup = EditSettingPopup()
        self._popup.bind(on_validate=self._validate)
        self._popup.bind(on_dismiss=self._dismiss)

    def _to_numtype(self, v):
        try:
            if self.type == 'float': return round(float(v), 1)
            else: return int(v)
        except ValueError:
            return self.min

    def _dismiss(self, *l):
        self._popup.ids.input.focus = False

    def _validate(self, instance, value):
        self._popup.dismiss()
        val = self._to_numtype(self._popup.ids.input.text)
        if val < self.min: val = self.min
        elif val > self.max: val = self.max
        self.value = val

    def on_touch_down(self, touch):
        if not self.ids.sliderlabel.collide_point(*touch.pos):
            return super(StdSettingSlider, self).on_touch_down(touch)
        ids = self._popup.ids
        ids.value = str(self.value)
        ids.input.text = str(self._to_numtype(self.value))
        self._popup.open()
        ids.input.focus = True
        ids.input.select_all()

class StdSettingSpinner(Spinner):
    pass

Factory.register('StdSettingsContainer', cls=StdSettingsContainer)
Factory.register('StdSettingTitle', cls=StdSettingTitle)
Factory.register('StdSettingBoolean', cls=StdSettingBoolean)
Factory.register('StdSettingSlider', cls=StdSettingSlider)
Factory.register('StdSettingString', cls=StdSettingString)
Factory.register('StdSettingStringLong', cls=StdSettingStringLong)
Factory.register('StdSettingSpinner', cls=StdSettingSpinner)
Factory.register('MyButton', cls=MyButton)
Factory.register('MyStrokeButton', cls=MyStrokeButton)
Factory.register('MySmoothButton', cls=MySmoothButton)
