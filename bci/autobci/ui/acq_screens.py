# coding: utf-8
import re
import os
import random
import time
import threading
import numpy as np
from kivy.clock import Clock
from datetime import datetime
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty, StringProperty, NumericProperty, ListProperty, BooleanProperty
from plyer import filechooser
from proc.sample_manager import SampleManager
from proc.utils import PATH_TO_SESSION
from proc.ds_converter import iii3a, iv2a, iv2b, lee19
# import math
# import pyautogui
# import pygame
# from pygame.locals import*
# import collections
# import kivy.garden.bar  # from kivy.garden.bar import Bar
# from kivy.app import App
# from kivy.uix.checkbox import CheckBox
# from kivy.uix.spinner import Spinner
# from kivy.uix.textinput import TextInput
# from kivy.uix.widget import Widget
# from kivy.core.window import Window
# import tkinter as tk
# from tkinter import filedialog

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class AcqSimu(Screen):
    subj_list = None
    ds_list = ('III 3a', 'IV 2a', 'IV 2b', 'LEE 19')  # sorted(os.listdir('/mnt/dados/eeg_data/eeg_epochs/'))
    selection = ListProperty([])
    selection2 = ListProperty([])

    def __init__(self, session, **kwargs):  # layout
        super(AcqSimu, self).__init__(**kwargs)
        self.session = session
        self.daisy = False
        self.new_dataset = None

    def back_to_bci(self, *args):
        self.manager.current = 'BCIMenu'
        self.manager.transition.direction = 'right'

    def change_to_acq_protocol(self, *args):
        # print(self.session.acq.daisy, self.session.acq.sample_rate, self.session.acq.flag, self.session.acq.mode,
        #     self.session.acq.com_port, self.session.acq.ch_labels, self.session.acq.path_to_file, self.session.acq.board)
        self.session.acq.mode = 'simu' # if self.ids.simulator.active else 'openbci'
        self.session.acq.sample_rate = self.ids.srate.value
        self.session.acq.ch_labels = self.ids.ch_labels.value
        self.session.acq.flag_mode = True

        # if self.session.acq.mode == 'simu':
        self.session.acq.board = None
        self.session.acq.com_port = None
        self.session.acq.daisy = None
        self.session.acq.ch_labels = self.ids.ch_labels.value
        # self.session.acq.class_ids = list(map(int, self.ids.class_ids.text.split(' ')))
        self.session.acq.class_ids = [1, 2]
        self.session.acq.eeg_path_simu_acq = self.ids.eeg_path.value
        self.session.acq.dummy = None # self.ids.dummy.value
        # else:
        #     self.session.acq.board = self.ids.acq_board.text
        #     self.session.acq.com_port = self.ids.com_port.value
        #     self.session.acq.daisy = self.ids.daisy.value
        #     self.session.acq.ch_labels = self.ids.ch_labels.value
        #     self.session.acq.class_ids = [1, 2]
        #     self.session.acq.eeg_path_simu_acq = None
        #     self.session.acq.dummy = None

        self.session.saveSession()
        self.manager.current = 'AcqProtocol'
        self.manager.transition.direction = 'left'

    def load_saved_settings(self): # carrega as preferencias de aquisição caso o usuário já tenha salvo alguma vez
        # print(self.session.acq.ch_labels)

        # if self.session.acq.mode == 'simu':
        # self.ids.item_acq_board.disabled = True

        if self.session.acq.mode == 'simu':
            self.ids.check_public.active = False
            self.ids.item_dataset.disabled = True
            self.ids.field_dataset.disabled = True
            self.ids.item_subject.disabled = True
            self.ids.field_subject.disabled = True
            self.ids.field_dataset_srate.disabled = True
            self.ids.eeg_dataset_path.disabled = True
            self.ids.path_to_original_dataset.disabled = True




            # self.ids.simulator.active = True
            self.ids.eeg_path.disabled = False
            self.ids.btn_explorer.disabled = False
            # self.ids.class_ids.disabled = False
            self.ids.ch_labels.disabled = False
            self.ids.srate.disabled = True
            self.ids.eeg_path.value = self.session.acq.eeg_path_simu_acq
            # self.ids.class_ids.text = '< Select >' if self.session.acq.class_ids is None else str(
            #     self.session.acq.class_ids[0]) + ' ' + str(self.session.acq.class_ids[1])
            self.ids.ch_labels.value = self.session.acq.ch_labels
            self.ids.srate.value = self.session.acq.sample_rate
            self.update_settings()

        # else:
        #     self.ids.simulator.active = False
        #     self.ids.eeg_path.disabled = True
        #     self.ids.btn_explorer.disabled = True
        #     self.ids.eeg_path.value = ''
        #
        #     self.ids.item_acq_board.disabled = False
        #     self.ids.acq_board.disabled = False
        #     self.ids.acq_board.text = self.session.acq.board
        #     self.ids.com_port.value = '' if self.session.acq.com_port is None else self.session.acq.com_port
        #     self.ids.daisy.value = self.session.acq.daisy
        #
        #     if self.ids.com_port.value == '' or self.ids.acq_board.text == '< Select >':
        #         self.ids.btn_next_step_acq.disabled = True
        #     else: self.ids.btn_next_step_acq.disabled = False
        #
        #     if self.ids.acq_board.text == '< Select >':
        #         self.ids.com_port.disabled = True
        #         self.ids.ch_labels.disabled = True
        #         self.ids.daisy.disabled = True
        #     else:
        #         self.ids.com_port.disabled = False
        #         self.ids.ch_labels.disabled = False
        #         self.ids.daisy.disabled = False

    def update_settings(self):
        self.ids.eeg_path.disabled = False
        self.ids.btn_explorer.disabled = False

        if self.ids.eeg_path.value != '':
            try:
                self.data, self.events, self.info = np.load(self.ids.eeg_path.value, allow_pickle=True) # load_pickle_data(self.ids.eeg_path.value)
                self.ids.srate.value = self.info['fs']
                self.ids.ch_labels.value = str(self.info['ch_labels'])
                self.ids.ch_labels.disabled = False
                # self.ids.item_class_ids.disabled = False
                # self.ids.class_ids.disabled = False

                # if self.info['class_ids'] == [1, 2, 3, 4]:
                #     self.ids.class_ids.values = ['< Select >', '1 2', '1 3', '1 4', '2 3', '2 4', '3 4']
                # elif self.info['class_ids'] == [2, 3]:
                #     self.ids.class_ids.values = ['< Select >', '2 3']
                #     self.ids.class_ids.text = '2 3'
                # else:
                #     self.ids.class_ids.values = ['< Select >', '1 2']
                #     self.ids.class_ids.text = '1 2'
            except:
                # self.ids.class_ids.text = '< Select >'
                # self.ids.item_class_ids.disabled = True
                # self.ids.class_ids.disabled = True
                self.ids.ch_labels.disabled = True
                self.ids.ch_labels.value = ''
                self.ids.srate.value = 250

        self.ids.btn_next_step_acq.disabled = True if (self.ids.eeg_path.value == '') else False # or self.ids.class_ids.text == '< Select >'

        # else:
        #     self.ids.eeg_path.disabled = True
        #     self.ids.btn_explorer.disabled = True
        #     self.ids.item_acq_board.disabled = False
        #     self.ids.acq_board.disabled = False
        #     self.ids.srate.value = 125 if self.ids.daisy.value else 250
        #     self.ids.btn_next_step_acq.disabled = True if (self.ids.com_port.value == '' or self.ids.acq_board.text == '< Select >') else False
        #
        #     if self.ids.acq_board.text == '< Select >':
        #         self.ids.com_port.disabled = True
        #         self.ids.ch_labels.disabled = True
        #         self.ids.daisy.disabled = True
        #     else:
        #         self.ids.com_port.disabled = False
        #         self.ids.ch_labels.disabled = False
        #         self.ids.daisy.disabled = False

    def enabled_publics(self):
        if self.ids.check_public.active:
            self.ids.item_dataset.disabled = False
            self.ids.field_dataset.disabled = False
            self.ids.item_subject.disabled = False
            self.ids.field_subject.disabled = False
            self.ids.field_dataset_srate.disabled = False
            self.ids.eeg_dataset_path.disabled = False
            self.ids.path_to_original_dataset.disabled = False
            self.ids.field_dataset_srate.disabled = True
        else:
            self.ids.item_dataset.disabled = True
            self.ids.field_dataset.disabled = True
            self.ids.item_subject.disabled = True
            self.ids.field_subject.disabled = True
            self.ids.field_dataset_srate.disabled = True
            self.ids.eeg_dataset_path.disabled = True
            self.ids.path_to_original_dataset.disabled = True
            self.ids.field_dataset_srate.disabled = True

    def try_load_dataset(self):
        if self.ids.eeg_dataset_path.value != '':
            # try:
            if self.ids.field_dataset.text == 'III 3a':
                self.new_dataset = iii3a(self.ids.eeg_dataset_path.value, self.ids.field_subject.text, self.ids.field_dataset.text)
            elif self.ids.field_dataset.text == 'IV 2a':
                self.new_dataset = iv2a(self.ids.eeg_dataset_path.value, self.ids.field_subject.text, self.ids.field_dataset.text)
                # self.data, self.events, self.info = np.load(self.ids.eeg_path.value, allow_pickle=True)  # load_pickle_data(self.ids.eeg_path.value)
            elif self.ids.field_dataset.text == 'IV 2b':
                self.new_dataset = iv2b(self.ids.eeg_dataset_path.value, self.ids.field_subject.text, self.ids.field_dataset.text)
            elif self.ids.field_dataset.text == 'IV 2a':
                self.new_dataset = lee19(self.ids.eeg_dataset_path.value, self.ids.field_subject.text, self.ids.field_dataset.text)
            self.ids.field_dataset_srate.value = self.new_dataset[2]['fs']
            self.ids.field_dataset_srate.disabled = True
            # except:
            #     self.new_dataset = None
            #     self.ids.btn_convert.disabled = True

        # print(self.ids.check_public.active,self.ids.eeg_dataset_path.value != '',self.ids.field_subject.text != '< Select >',self.ids.field_dataset.text != '< Select >')
        self.ids.btn_convert.disabled = False if (self.ids.check_public.active and self.ids.eeg_dataset_path.value != '' and self.ids.field_subject.text != '< Select >' and self.ids.field_dataset.text != '< Select >') else True

    def update_subject_values(self, *args): # chamada ao selecionar um dataset no field_dataset
        if self.ids.field_dataset.text == 'III 3a':
            self.subj_list = ['K3', 'K6', 'L1']
            self.ids.field_dataset_srate.value = 250
        # elif self.ids.field_dataset.text == 'III 4a':
        #     self.subj_list = ['aa', 'al', 'av', 'aw', 'ay']
        #     self.ids.field_dataset_srate.value = 100
        elif self.ids.field_dataset.text in ['IV 2a', 'IV 2b']:
            self.subj_list = list(map(lambda x: str(x), np.arange(1, 10)))
            self.ids.field_dataset_srate.value = 250
        elif self.ids.field_dataset.text in ['LEE 19']:
            self.subj_list = list(map(lambda x: str(x), np.arange(1, 55)))
            self.ids.field_dataset_srate.value = 125
        else:
            self.subj_list = ['--']
            self.ids.field_dataset_srate.value = 250
        self.ids.field_subject.values = self.subj_list

    def convert_dataset(self):
        path_out = ''
        if not self.new_dataset is None:
            path = self.ids.eeg_dataset_path.value
            # try:
            if self.ids.field_dataset.text == 'III 3a': prefix = ''; sufix = ''
            elif self.ids.field_dataset.text == 'IV 2a': prefix = 'A0'; sufix = 'E' if 'E.gdf' in path else 'T'
            elif self.ids.field_dataset.text == 'IV 2b': prefix = 'B0'; sufix = '01T' if '01T' in path else '02T' if '02T' in path else '03T' if '03T' in path else '04T' if '04T' in path else '05T'
            elif self.ids.field_dataset.text == 'LEE 19': prefix = 'S'; sufix = '_sess02' if 'sess02_' in path else '_sess01'
            path_out = os.path.dirname(path) + '/' + prefix + str(self.ids.field_subject.text) + sufix
            np.save(path_out, self.new_dataset, allow_pickle=True)
            path_out += '.npy'
            self.popup_convert_sucess()
            # except:
            #     path_out = ''
            #     self.popup_convert_error()

            self.ids.eeg_path.value = path_out
            self.clear_fields()
            self.update_settings()

    def popup_convert_sucess(self, *args):
        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
        pop = Popup(title='Aviso', content=box, size_hint=(None, None), size=(300, 180))
        label = Label(text='Arquivo convertido com sucesso para o formato .npy.')
        botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
        box.add_widget(label)
        box.add_widget(botao)
        pop.open()

    def popup_convert_error(self, *args):
        self.session.saveSession()
        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
        pop = Popup(title='Aviso', content=box, size_hint=(None, None), size=(300, 180))
        label = Label(text='Erro ao converter arquivo.')
        botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
        box.add_widget(label)
        box.add_widget(botao)
        pop.open()

    def clear_fields(self, *args):
        self.ids.eeg_dataset_path.value = ''
        self.ids.field_dataset.text = '< Select >'
        self.ids.field_subject.text = '< Select >'
        self.ids.field_dataset_srate.value = 250
        self.ids.field_dataset_srate.disabled = True
        self.ids.check_public.active = False
        self.enabled_publics()

    # def choose(self):
    #     root = tk.Tk(); root.withdraw(); return (filedialog.askopenfilename())

    def choose_eeg_path(self): # Call plyer filechooser API
        filechooser.open_file(on_selection=self.handle_selection)

    def handle_selection(self, selection): # Callback function for handling the selection response
        self.selection = selection

    def on_selection(self, *a, **k): # Update TextInput.text after FileChoose.selection is changed via FileChoose.handle_selection
        self.ids.input_eeg_path.text = str(self.selection).replace('[', '').replace(']', '').replace("'", '')

    def choose_eeg_ds_path(self):
        filechooser.open_file(on_selection=self.handle_selection2)

    def handle_selection2(self, selection):
        self.selection2 = selection

    def on_selection2(self, *a, **k):
        self.ids.input_eeg_ds_path.text = str(self.selection2).replace('[', '').replace(']', '').replace("'", '')

# class Menu(GridLayout):
#     pass
#
# class SettingsScreens(ScreenManager):
#     simulator = ObjectProperty(None)
#     openbci = ObjectProperty(None)
#
# class Simulator(Screen):
#     eeg_path = StringProperty('')
#     labels_path = StringProperty('')
#     srate = NumericProperty(0)
#
# class OpenBCI(Screen):
#     pass


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class AcqReal(Screen):
    subj_list = None
    selection = ListProperty([])
    selection2 = ListProperty([])

    def __init__(self, session, **kwargs):  # layout
        super(AcqReal, self).__init__(**kwargs)
        self.session = session
        self.daisy = False

    def back_to_bci(self, *args):
        self.manager.current = 'BCIMenu'
        self.manager.transition.direction = 'right'

    def change_to_acq_protocol(self, *args):
        # print(self.session.acq.daisy, self.session.acq.sample_rate, self.session.acq.flag, self.session.acq.mode,
        #     self.session.acq.com_port, self.session.acq.ch_labels, self.session.acq.path_to_file, self.session.acq.board)
        self.session.acq.mode = 'openbci'
        self.session.acq.sample_rate = self.ids.srate.value
        self.session.acq.flag_mode = True

        # if self.session.acq.mode == 'simu':
        #     self.session.acq.board = None
        #     self.session.acq.com_port = None
        #     self.session.acq.daisy = None
        #     self.session.acq.ch_labels = self.ids.ch_labels.value
        #     self.session.acq.class_ids = list(map(int, self.ids.class_ids.text.split(' ')))
        #     self.session.acq.eeg_path_simu_acq = self.ids.eeg_path.value
        #     self.session.acq.dummy = None # self.ids.dummy.value
        # else:
        self.session.acq.board = self.ids.acq_board.text
        self.session.acq.com_port = self.ids.com_port.value
        # self.session.acq.daisy = self.ids.daisy.value
        self.session.acq.daisy = self.ids.daisy.active
        self.session.acq.ch_labels = self.ids.ch_labels.value
        self.session.acq.class_ids = [1, 2]
        self.session.acq.eeg_path_simu_acq = None
        self.session.acq.dummy = None

        # self.ids.check_public.active = False
        self.session.saveSession()
        self.manager.current = 'AcqProtocol'
        self.manager.transition.direction = 'left'

    def load_saved_settings(self): # carrega as preferencias de aquisição caso o usuário já tenha salvo alguma vez
        # print(self.session.acq.ch_labels)
        if self.session.acq.mode == 'openbci':
            self.ids.ch_labels.value = self.session.acq.ch_labels
            self.ids.srate.value = self.session.acq.sample_rate
            # if self.session.acq.mode == 'simu':
            #     self.ids.item_acq_board.disabled = True
            #     self.ids.acq_board.disabled = True
            #     self.ids.acq_board.text = '< Select >'
            #     self.ids.com_port.disabled = True
            #     self.ids.ch_labels.disabled = True
            #     self.ids.daisy.disabled = True
            #
            #     self.ids.simulator.active = True
            #     self.ids.eeg_path.disabled = False
            #     self.ids.btn_explorer.disabled = False
            #     self.ids.class_ids.disabled = False
            #     self.ids.eeg_path.value = self.session.acq.eeg_path_simu_acq
            #     self.ids.class_ids.text = '< Select >' if self.session.acq.class_ids is None else str(
            #         self.session.acq.class_ids[0]) + ' ' + str(self.session.acq.class_ids[1])
            #     self.update_settings()
            #
            # else:
            # self.ids.simulator.active = False
            # self.ids.eeg_path.disabled = True
            # self.ids.btn_explorer.disabled = True
            # self.ids.eeg_path.value = ''

            self.ids.item_acq_board.disabled = False
            self.ids.acq_board.disabled = False
            self.ids.acq_board.text = self.session.acq.board
            self.ids.com_port.value = '' if self.session.acq.com_port is None else self.session.acq.com_port
            # self.ids.daisy.value = self.session.acq.daisy
            self.ids.daisy.active = self.session.acq.daisy

            if self.ids.com_port.value == '' or self.ids.acq_board.text == '< Select >':
                self.ids.btn_next_step_acq.disabled = True
            else: self.ids.btn_next_step_acq.disabled = False

            if self.ids.acq_board.text == '< Select >':
                self.ids.com_port.disabled = True
                self.ids.ch_labels.disabled = True
                self.ids.daisy.disabled = True
                self.ids.item_daisy.disabled = True
            else:
                self.ids.com_port.disabled = False
                self.ids.ch_labels.disabled = False
                self.ids.daisy.disabled = False
                self.ids.item_daisy.disabled = False

    def update_settings(self):
        # if self.ids.simulator.active:
        #     self.ids.eeg_path.disabled = False
        #     self.ids.btn_explorer.disabled = False
        #     self.ids.item_acq_board.disabled = True
        #     self.ids.acq_board.disabled = True
        #     self.ids.acq_board.text = '< Select >'
        #     self.ids.com_port.disabled = True
        #     self.ids.ch_labels.disabled = True
        #     self.ids.daisy.disabled = True
        #
        #     if self.ids.eeg_path.value != '':
        #         try:
        #             self.data, self.events, self.info = np.load(self.ids.eeg_path.value, allow_pickle=True) # load_pickle_data(self.ids.eeg_path.value)
        #             self.ids.srate.value = self.info['fs']
        #             self.ids.ch_labels.value = str(self.info['ch_labels'])
        #             self.ids.item_class_ids.disabled = False
        #             self.ids.class_ids.disabled = False
        #             # print(self.data.shape)
        #             if self.info['class_ids'] == [1, 2, 3, 4]:
        #                 self.ids.class_ids.values = ['< Select >', '1 2', '1 3', '1 4', '2 3', '2 4', '3 4']
        #             elif self.info['class_ids'] == [2, 3]:
        #                 self.ids.class_ids.values = ['< Select >', '2 3']
        #                 self.ids.class_ids.text = '2 3'
        #             else:
        #                 self.ids.class_ids.values = ['< Select >', '1 2']
        #                 self.ids.class_ids.text = '1 2'
        #         except:
        #             self.ids.class_ids.text = '< Select >'
        #             self.ids.item_class_ids.disabled = True
        #             self.ids.class_ids.disabled = True
        #
        #     self.ids.btn_next_step_acq.disabled = True if (self.ids.eeg_path.value == '' or self.ids.class_ids.text == '< Select >') else False

        # else:
        # self.ids.eeg_path.disabled = True
        # self.ids.btn_explorer.disabled = True
        self.ids.item_acq_board.disabled = False
        self.ids.acq_board.disabled = False
        # self.ids.srate.value = 125 if self.ids.daisy.value else 250
        self.ids.srate.value = 125 if self.ids.daisy.active else 250
        self.ids.btn_next_step_acq.disabled = True if (self.ids.com_port.value == '' or self.ids.acq_board.text == '< Select >') else False

        if self.ids.acq_board.text == '< Select >':
            self.ids.com_port.disabled = True
            self.ids.ch_labels.disabled = True
            self.ids.daisy.disabled = True
            self.ids.item_daisy.disabled = True
        else:
            self.ids.com_port.disabled = False
            self.ids.ch_labels.disabled = False
            self.ids.daisy.disabled = False
            self.ids.item_daisy.disabled = False

    def choose(self):
        filechooser.open_file(on_selection=self.handle_selection)

    def choose2(self):
        filechooser.open_file(on_selection=self.handle_selection2)


    def handle_selection(self, selection):
        self.selection = selection

    def handle_selection2(self, selection):
        self.selection2 = selection

    def on_selection(self, *a, **k):
        self.ids.input.text = str(self.selection).replace('[', '').replace(']', '').replace("'", '')

    def on_selection2(self, *a, **k):
        self.ids.input2.text = str(self.selection2).replace('[', '').replace(']', '').replace("'", '')

# class Menu(GridLayout):
#     pass
#
# class SettingsScreens(ScreenManager):
#     simulator = ObjectProperty(None)
#     openbci = ObjectProperty(None)
#
# class Simulator(Screen):
#     eeg_path = StringProperty('')
#     labels_path = StringProperty('')
#     srate = NumericProperty(0)
#
# class OpenBCI(Screen):
#     pass

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class AcqProtocol(Screen):
    def __init__(self, session, **kwargs):  # layout
        super(AcqProtocol, self).__init__(**kwargs)
        self.session = session
        self.daisy = False

    def back_to_mode(self, *args):

        if self.session.acq.mode == 'simu': self.manager.current = 'AcqSimu'
        else: self.manager.current = 'AcqReal'

        self.manager.transition.direction = 'right'

    def run_acq(self, *args):
        self.session.acq.n_runs = self.ids.n_runs.value  # Número de Execuções - conjunto de tentativas
        self.session.acq.runs_interval = self.ids.runs_interval.value  # Intervalo de Tempo em segundos entre execuções
        self.session.acq.n_trials = self.ids.n_trials.value  # Número de tentativas a serem exibidas por execução (deve ser PAR)
        self.session.acq.cue_offset = self.ids.cue_offset.value  # Momento para o início da apresentação da dica (s) (a partir do início da tentativa)
        self.session.acq.cue_time = self.ids.cue_time.value  # Duração da apresentação da dica (s)
        self.session.acq.min_pause = self.ids.pause_min.value  # Pausa entre tentativas (s)
        self.session.acq.trial_duration = self.ids.trial_duration.value  # Duração total da tentativa (s)
        self.session.acq.flag_protocol = True
        self.session.saveSession()
        self.manager.current = 'AcqRun'
        self.manager.transition.direction = 'left'

    def update_settings(self):
        # atualiza a sessão do usuário ativo com as informações oriundas do formulário da UI acquisition_settings.kv
        self.ids.n_runs.value = self.session.acq.n_runs
        self.ids.runs_interval.value = self.session.acq.runs_interval
        self.ids.n_trials.value = self.session.acq.n_trials
        self.ids.cue_offset.value = self.session.acq.cue_offset
        self.ids.cue_time.value = self.session.acq.cue_time
        self.ids.pause_min.value = self.session.acq.min_pause
        self.ids.trial_duration.value = self.session.acq.trial_duration

    def update_trial_duration(self):
        self.ids.trial_duration.value = self.ids.cue_offset.value + self.ids.cue_time.value + self.ids.pause_min.value + 1


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class AcqRun(Screen):
    src = ["ui/resources/cross2.png",
           "ui/resources/left2.png",
           "ui/resources/right2.png",
           "ui/resources/blank.png",
           "ui/resources/break.png"]
    fig_list = ListProperty(src)
    button_stream = StringProperty('Iniciar experimento')
    carousel = ObjectProperty(None)
    inst_prob_left = NumericProperty(0)
    inst_prob_right = NumericProperty(0)
    accum_color_left = ListProperty([1, 0, 0, 1])
    accum_color_right = ListProperty([0, 0, 1, 1])

    def __init__(self, session, **kwargs):
        super(AcqRun, self).__init__(**kwargs)
        self.session = session
        self.carousel.index = 3  # 3 == ui/resources/blank.png
        self.stream_flag = False  # indica status do streaming (True = Operando; False = Parado)

    def back_to_acq(self, *args):
        self.manager.current = 'AcqProtocol'
        self.manager.transition.direction = 'right'

    def toggle_stream(self, *args):  #### step 1 - after press on Start Streaming ## chaveia o streaming
        if self.stream_flag:  # se True
            self.stream_stop()
            self.stop_stimulus()
        else:
            self.stream_start()

    def stream_start(self):  ### step 2
        # print(self.session.acq.dummy)
        self.sm = SampleManager(self.session.acq.sample_rate,
                                self.session.acq.com_port,
                                # self.session.dp.epoch_len,
                                class_ids=self.session.acq.class_ids,
                                daisy=self.session.acq.daisy,
                                mode=self.session.acq.mode,
                                path=self.session.acq.eeg_path_simu_acq,
                                # labels_path=self.session.acq.path_to_labels_file,
                                # dummy=self.session.acq.daisy
                                )

        # print(self.sm.loadedData)
        # self.sm.daemon = True
        self.sm.stop_flag = False
        self.sm.start()  # inicia thread - chama def run() em sample_manager.py
        self.button_stream = 'Parar experimento'  # seta o label do botão na tela cal_start 'Parar Transmiss\u00e3o'
        self.stream_flag = True
        self.start_stimulus()
        self.set_bar_default()

    def stream_stop(self):
        self.sm.stop_flag = True
        self.stream_flag = False
        self.sm.join()  ### encadeamento de chamada fica bloqueado até o objeto de encadeamento no qual foi chamado terminar
        self.button_stream = 'Iniciar experimento'
        self.set_bar_default()
        self.acq_info = {
            'fs': self.session.acq.sample_rate,
            'class_ids': list(np.unique(self.stim_list)),
            'trial_tcue': self.session.acq.cue_offset,
            'trial_tpause': self.session.acq.cue_offset + self.session.acq.cue_time,
            'trial_mi_time': self.session.acq.cue_time,
            'trials_per_class': (self.session.acq.n_trials * self.session.acq.n_runs) / 2,
            'eeg_channels': self.sm.loadedData.shape[0],
            'ch_labels': self.session.acq.ch_labels,
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')
        }

        # self.save_data()  # save 1
        save_acq = AcqSavePopup(self.session, self.sm, self.acq_info)
        save_acq.open() # save 2, Popup mode

    def start_stimulus(self):
        self.epoch_counter = 0
        self.run_counter = 0
        self.generate_stim_list()
        self.start_run(None)

    def generate_stim_list(self):
        nt = self.session.acq.n_trials * self.session.acq.n_runs
        idA = self.session.acq.class_ids[0] * np.ones(int(nt / 2))  # original sem o int
        idB = self.session.acq.class_ids[1] * np.ones(int(nt / 2))  # original sem o int
        slist = np.concatenate([idA, idB])
        random.shuffle(slist)  # embaralha vetor
        self.stim_list = slist.astype(int)

    def stop_stimulus(self):
        Clock.unschedule(self.display_epoch)
        Clock.unschedule(self.start_run)
        Clock.unschedule(self.set_pause)
        Clock.unschedule(self.set_cue)
        Clock.unschedule(self.set_blank)
        self.carousel.index = 3

    def start_run(self, dt):
        self.run_epoch_counter = 0
        self.carousel.index = 3  # 3 para manipular as etapas e imagens do protocolo
        Clock.schedule_interval(self.display_epoch, self.session.acq.trial_duration)

    def stop_run(self):
        self.stop_stimulus()
        self.run_counter += 1
        if self.run_counter < self.session.acq.n_runs:
            Clock.schedule_once(self.start_run, self.session.acq.runs_interval)
            self.carousel.index = 4
        else:
            self.stream_stop()
            self.stop_stimulus()

    def display_epoch(self, dt):
        st = time.time()
        if self.run_epoch_counter < self.session.acq.n_trials:
            Clock.schedule_once(self.set_start_trial)
            Clock.schedule_once(self.set_cue, self.session.acq.cue_offset)
            Clock.schedule_once(self.set_blank, self.session.acq.cue_offset + self.session.acq.cue_time)
            #self.pause_time = self.session.acq.min_pause # random.uniform(self.session.acq.min_pause, self.session.acq.min_pause + 2)
            #Clock.schedule_once(self.set_pause, self.session.acq.cue_offset + self.session.acq.cue_time + self.pause_time)
        else:
            self.stop_run()

    def set_start_trial(self, dt):
        self.carousel.index = 0  # original comentado
        self.sm.MarkEvents(0)
        self.beep()

    def set_pause(self, dt):
        # print(self.session.acq.cue_offset, self.session.acq.cue_time, self.pause_time)
        pass

    def set_cue(self, dt):
        if self.stim_list[self.epoch_counter] == self.session.acq.class_ids[0]:
            self.carousel.index = self.session.acq.class_ids[0]  # original comentado
            #self.sm.event_list[-1, 1] = 1
            self.sm.MarkEvents(self.session.acq.class_ids[0]) #101
            anim_left = threading.Thread(target=self.animate_bar_left)
            anim_left.start()
        elif self.stim_list[self.epoch_counter] == self.session.acq.class_ids[1]:
            self.carousel.index = self.session.acq.class_ids[1]  # original comentado
            #self.sm.event_list[-1, 1] = 2
            self.sm.MarkEvents(self.session.acq.class_ids[1]) #102
            anim_right = threading.Thread(target=self.animate_bar_right)
            anim_right.start()
        self.epoch_counter += 1
        self.run_epoch_counter += 1

    def set_blank(self, dt):
        self.carousel.index = 3

    def beep(self):
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f &' % (0.3, 500))

    def save_data(self):
        print('Salvando dados')
        acq_user_path = PATH_TO_SESSION + self.session.info.nickname + '/acqs/'
        if not os.path.isdir(acq_user_path): os.makedirs(acq_user_path)
        self.sm.SaveAll(self.acq_info, acq_user_path + 'data')
        # self.sm.SaveData(acq_user_path + 'data')
        # self.sm.SaveEvents(acq_user_path + 'events')
        self.update_user_path(PATH_TO_SESSION + self.session.info.nickname + '/acqs/')

    def update_user_path(self, path):
        pattern_data = re.compile(r"data_.*?\.npy")  # mask to get filenames
        pattern_events = re.compile(r"events_.*?\.npy")  # mask to get filenames
        data_files, events_files = [], []
        for root, dirs, files in os.walk(path):  # add .kv files of ui/kv in kv_files vector
            data_files += [root + file_ for file_ in files if pattern_data.match(file_)]
            events_files += [root + file_ for file_ in files if pattern_events.match(file_)]
        try:
            last_data_file = max(data_files, key=os.path.getmtime).split('/')[-1]
            last_ev_file = max(events_files, key=os.path.getmtime).split('/')[-1]
            self.session.acq.path_to_eeg_data = path + last_data_file
            self.session.acq.path_to_eeg_events = path + last_ev_file
        except:
            self.session.acq.path_to_eeg_data = None
            self.session.acq.path_to_eeg_events = None
        self.session.saveSession()



    def set_bar_default(self):
        self.inst_prob_left = 50
        self.inst_prob_right = 50
        self.accum_color_left = [1, 0, 0, 1]
        self.accum_color_right = [0, 0, 1, 1]

    def animate_bar_left(self):
        ts = time.time()
        tf = 0
        while tf < self.session.acq.cue_time and self.stream_flag:
            ratio = tf / self.session.acq.cue_time
            self.inst_prob_left = 50 + ratio * 50
            # if self.inst_prob_left > 80: self.accum_color_left = [1, 1, 0, 1]
            # else: self.accum_color_left = [1, 0, 0, 1]
            self.inst_prob_right = 100 - self.inst_prob_left
            time.sleep(0.05)
            tf = (time.time() - ts)
        self.set_bar_default()

    def animate_bar_right(self):
        ts = time.time()
        tf = 0
        while tf < self.session.acq.cue_time and self.stream_flag:
            ratio = tf / self.session.acq.cue_time
            self.inst_prob_right = 50 + ratio * 50
            # if self.inst_prob_right > 80: self.accum_color_right = [1, 1, 0, 1]
            # else: self.accum_color_right = [0, 0, 1, 1]
            self.inst_prob_left = 100 - self.inst_prob_right
            time.sleep(0.05)
            tf = (time.time() - ts)
        self.set_bar_default()

class AcqSavePopup(Popup):
    def __init__(self, session, sm, acq_info, **kwargs):
        super(AcqSavePopup, self).__init__(**kwargs)
        self.session = session
        self.sm = sm
        self.acq_info = acq_info
        print('Salvando dados..')

    def save_acquisition(self, save_name):
        acq_user_path = PATH_TO_SESSION + self.session.info.nickname + '/acqs/'
        if not os.path.isdir(acq_user_path): os.makedirs(acq_user_path)
        self.sm.SaveAll(self.acq_info, acq_user_path + save_name)
        # self.sm.SaveData(acq_user_path + save_name + '_data')
        # self.sm.SaveEvents(acq_user_path + save_name + '_events')
        self.session.acq.path_to_eeg_data = acq_user_path + save_name + '_data.npy'
        self.session.acq.path_to_eeg_events = acq_user_path + save_name + '_events.npy'
        self.session.saveSession()
        print('Dados armazenados no disco!')


