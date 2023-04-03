# coding: utf-8
import numpy as np
from plyer import filechooser
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import Screen
from kivy.properties import NumericProperty, ListProperty, ObjectProperty, StringProperty, BooleanProperty
from proc.utils import PATH_TO_SESSION, FONT_SIZE
from proc.processor import Processor
from proc.optimizer import AutoSetup
# from ui.template import StdSettingItem

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class CalLoad(Screen):
    max_channels = NumericProperty(8)
    selection = ListProperty([])
    selection2 = ListProperty([])

    def __init__(self, session, **kwargs):
        super(CalLoad, self).__init__(**kwargs)
        self.session = session

    def back_to_bci(self, *args):
        self.manager.current = 'BCIMenu'
        self.manager.transition.direction = 'right'

    def change_to_cal_setup(self, *args):
        self.manager.current = 'CalSettings'
        self.manager.transition.direction = 'left'

    def popup_required(self, *args):
        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
        pop = Popup(title='Attention', content=box, size_hint=(None, None), size=(500, 180))
        label = Label(text='You need to set all eeg data info:\nEEG file, sample rate, class_ids and channels!')
        botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
        box.add_widget(label)
        box.add_widget(botao)
        pop.open()

    def popup_load_error(self, *args):
        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
        pop = Popup(title='Attention', content=box, size_hint=(None, None), size=(300, 180))
        label = Label(text='EEG file not found!')
        botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
        box.add_widget(label)
        box.add_widget(botao)
        pop.open()

    def popup_channels_error(self, no_channels, max_channels, *args):
        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
        pop = Popup(title='Attention', content=box, size_hint=(None, None), size=(500, 180))
        label = Label(text='Indexes ' + str(list(map(lambda x: x+1, no_channels))) + ' are outside the range of valid channels.\nIndex must be between 1 and ' + str(max_channels) + '.')
        botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
        box.add_widget(label)
        box.add_widget(botao)
        pop.open()

    def save_and_progress(self, *args):
        if self.ids.eeg_path == '' or self.ids.srate.value == 0 or self.ids.channels.value == '': # or self.ids.class_ids.text == '< Select >'
            self.popup_required()
        else:
            try:
                self.data, self.events, self.info = np.load(self.ids.eeg_path.value, allow_pickle=True)  # load_pickle_data() # load_mat_data()
                self.session.dp.eeg_path_train = self.ids.eeg_path.value
                self.session.dp.eeg_info = self.info
                # self.session.dp.class_ids = list(map(int, self.ids.class_ids.text.split(' ')))
                # self.session.dp.class_ids = list(map(int, '1 2'.split(' ')))
                self.session.dp.class_ids = [1, 2]
                self.session.dp.cross_val = self.ids.cross_val.active
                self.session.dp.n_folds = self.ids.n_folds.value if self.session.dp.cross_val else None
                self.session.dp.test_perc = self.ids.test_perc.value

                if not self.ids.cross_val.active:
                    self.data2, self.events2, self.info2 = np.load(self.ids.eeg_path2.value, allow_pickle=True)  # load_pickle_data() # load_mat_data()
                    self.session.dp.eeg_path_test = self.ids.eeg_path2.value
                else:
                    if self.ids.eeg_path2.value != '':
                        self.data2, self.events2, self.info2 = np.load(self.ids.eeg_path2.value, allow_pickle=True)  # load_pickle_data() # load_mat_data()
                        self.session.dp.eeg_path_test = self.ids.eeg_path2.value

                if ':' in self.ids.channels.value: # if :E or S:E (channels sequence)
                    limits = list(map(int, [ele for ele in self.ids.channels.value.split(':') if ele.isnumeric()]))
                    if len(limits) == 1: ch_idx = list(np.arange(0, limits[0])) # if :E
                    else: ch_idx = list(np.arange(limits[0]-1, limits[1])) # if S:E
                elif '-' in self.ids.channels.value: # se -1 (all channels)
                    # ch_idx = list(map(int, self.ids.channels.value.split(' ')))
                    ch_idx = list(np.arange(0, self.info['eeg_channels']))
                else: # if A B C D (channels list)
                    idx_list = [ele for ele in self.ids.channels.value.split(' ') if ele.isnumeric()]
                    if idx_list: ch_idx = list(map(lambda x: int(x) - 1, idx_list))
                    else: ch_idx = list(np.arange(0, self.info['eeg_channels']))

                # print(ch_idx)
                no_channels = [ele for ele in ch_idx if ele >= self.data.shape[0] or ele < 0]
                if no_channels != []: self.popup_channels_error(no_channels, self.data.shape[0])
                else:
                    self.session.dp.channels = ch_idx
                    self.session.dp.flag_load = True
                    self.session.saveSession()
                    self.change_to_cal_setup()
            except:
                self.popup_load_error()

    def set_disabled_field(self):
        self.ids.srate.value == 0
        # self.ids.class_ids.values = ''
        # self.ids.class_ids.text = '< Select >'
        self.ids.channels.value = '-1'
        self.ids.btn_save_progress.disabled = True
        # self.ids.item_class_ids.disabled = True
        # self.ids.class_ids.disabled = True
        self.ids.channels.disabled = True
        # self.ids.item_cross_val.disabled = True
        # self.ids.cross_val.disabled = True
        self.ids.test_perc.disabled = True
        self.ids.n_folds.disabled = False

    def set_enabled_field(self):
        self.ids.btn_save_progress.disabled = False
        # self.ids.item_class_ids.disabled = False
        # self.ids.class_ids.disabled = False
        self.ids.channels.disabled = False
        self.ids.item_cross_val.disabled = False
        self.ids.cross_val.disabled = False
        self.ids.test_perc.disabled = False if self.ids.cross_val.active else True
        self.ids.n_folds.disabled = False if self.ids.cross_val.active else True

    def update_settings(self):
        loaded_train, loaded_test = False, False
        if self.ids.eeg_path.value != '':
            try:
                self.data, self.events, self.info = np.load(self.ids.eeg_path.value, allow_pickle=True) # load_pickle_data() # load_mat_data()
                self.ids.srate.value = self.info['fs']
                self.max_channels = self.data.shape[0] # self.info['eeg_channels']
                loaded_train = True
                # if self.info['class_ids'] == [1, 2, 3, 4]:
                #     self.ids.class_ids.values = ['< Select >', '1 2', '1 3', '1 4', '2 3', '2 4', '3 4']
                # elif self.info['class_ids'] == [1, 3]:
                #     self.ids.class_ids.values = ['< Select >', '1 3']
                #     self.ids.class_ids.text = '1 3'
                # else:
                #     self.ids.class_ids.values = ['< Select >', '1 2']
                #     self.ids.class_ids.text = '1 2'
            except: self.set_disabled_field()

        if self.ids.eeg_path2.value != '':
            try:
                self.data2, self.events2, _ = np.load(self.ids.eeg_path2.value, allow_pickle=True)  # load_pickle_data() # load_mat_data()
                loaded_test = True
            except: pass

        if not loaded_train: self.set_disabled_field()
        else:
            if loaded_test: self.set_enabled_field()
            else:
                if self.ids.cross_val.active: self.set_enabled_field()
                else: self.set_disabled_field()

    def load_saved_settings(self, *args):
        self.ids.eeg_path.value = '' if self.session.dp.eeg_path_train is None else self.session.dp.eeg_path_train
        self.ids.eeg_path2.value = '' if self.session.dp.eeg_path_test is None else self.session.dp.eeg_path_test
        self.update_settings()
        # self.ids.class_ids.text = '< Select >' if self.session.dp.class_ids is None else str(self.session.dp.class_ids[0]) + ' ' + str(self.session.dp.class_ids[1]) # str(self.session.dp.class_ids).replace(',', '').replace('[', '').replace(']', '')
        self.ids.cross_val.active = self.session.dp.cross_val
        self.ids.n_folds.value = 10 if self.session.dp.n_folds is None else self.session.dp.n_folds
        self.ids.test_perc.value = 0.5 if self.session.dp.test_perc is None else self.session.dp.test_perc

        self.ids.item_cross_val.disabled = False
        self.ids.cross_val.disabled = False

        if self.session.dp.channels is None: self.ids.channels.value = '-1'
        else:
            channels_content = list(map(lambda x: int(x) + 1, self.session.dp.channels))
            # self.ids.channels.value = str(channels_content).replace(',', '').replace('[', '').replace(']', '')
            if len(channels_content) == self.max_channels: # self.session.dp.eeg_info['eeg_channels']
                self.ids.channels.value = '-1'
            else:
                ch_idx = ''
                for ele in list(map(str, channels_content)):
                    if ele.isnumeric():
                        ch_idx += (ele + ' ')
                self.ids.channels.value = ch_idx

    def check_crossval_enabled(self, *args):
        self.ids.n_folds.disabled = False if self.ids.cross_val.active and not self.ids.cross_val.disabled else True
        self.ids.test_perc.disabled = False if self.ids.cross_val.active and not self.ids.cross_val.disabled else True

    # def choose(self):
    #     root = tk.Tk(); root.withdraw(); return (filedialog.askopenfilename())

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
        self.ids.input2.text = str(self.selection2).replace('[', '').replace(']', '').replace("'", '')

    def on_selection2(self, *a, **k):
        self.ids.input2.text = str(self.selection2).replace('[', '').replace(']', '').replace("'", '')


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class CalSettings(Screen):
    epoch_lim_max = NumericProperty(8)
    epoch_lim_min = NumericProperty(8)
    srate = NumericProperty(8)

    def __init__(self, session, **kwargs):
        super(CalSettings, self).__init__(**kwargs)
        self.session = session
        self.acc_cal, self.acc_val = None, None
        self.kpa_cal, self.kpa_val = None, None
        self.cross_val = None

    def back_to_calLoad(self, *args):
        self.manager.current = 'CalLoad'
        self.manager.transition.direction = 'right'

    def calibrate(self, *args):
        self.save_config()
        if self.ids.check_auto_cal.active:
            auto_setup = AutoSetup(self.session, self.ids.load_last_setup.active)
            auto_setup.run_optimizer()
            self.save_opt_config(auto_setup.best) # , auto_setup.trials.argmin
            self.cross_val = False
            self.acc_cal = round((-1) * auto_setup.trials.best_trial['result']['loss'] * 100, 2)
            print(auto_setup.best) # auto_setup.trials.argmin # auto_setup.trials.best_trial['misc']['vals']
        else:
            setup = Processor(self.session)
            setup.define_params(f_low=self.session.dp.f_low, f_high=self.session.dp.f_high, ncsp=self.session.dp.ncsp,
                                class_ids=self.session.dp.class_ids, tmin=self.session.dp.epoch_start, tmax=self.session.dp.epoch_end,
                                fs=self.session.dp.eeg_info['fs'], filt_type=self.session.dp.filt_design, filt_order=self.session.dp.filt_order,
                                clf_dict=self.session.dp.final_clf, is_sbcsp=self.session.dp.is_sbcsp, nbands=self.session.dp.n_sbands,
                                overlap=self.session.dp.sb_overlap, crossval=False)
            setup.load_eeg_train(eeg_path=self.session.dp.eeg_path_train, channels=self.session.dp.channels)
            setup.load_eeg_test(eeg_path=self.session.dp.eeg_path_train, channels=self.session.dp.channels)
            setup.process()
            self.acc_cal = round(setup.acc*100,2)
            self.cross_val = self.session.dp.cross_val

        # print(self.session.dp.is_sbcsp, self.session.dp.eeg_info['fs'], self.session.dp.f_low, self.session.dp.f_high,
        #       self.session.dp.ncsp, self.session.dp.class_ids, self.session.dp.epoch_start, self.session.dp.epoch_end,
        #       self.session.dp.filt_design, self.session.dp.sb_clf, self.session.dp.final_clf, self.session.dp.filt_order,
        #       self.session.dp.n_sbands, self.session.dp.sb_overlap, self.session.dp.cross_val, self.session.dp.n_folds, self.session.dp.test_perc)

        setup = Processor(self.session)
        setup.load_eeg_train(eeg_path=self.session.dp.eeg_path_train, channels=self.session.dp.channels)
        setup.load_eeg_test(eeg_path=self.session.dp.eeg_path_test, channels=self.session.dp.channels)
        setup.define_params(f_low=self.session.dp.f_low, f_high=self.session.dp.f_high, ncsp=self.session.dp.ncsp,
                            class_ids=self.session.dp.class_ids, tmin=self.session.dp.epoch_start, tmax=self.session.dp.epoch_end,
                            fs=self.session.dp.eeg_info['fs'], filt_type=self.session.dp.filt_design, filt_order=self.session.dp.filt_order,
                            clf_dict=self.session.dp.final_clf, is_sbcsp=self.session.dp.is_sbcsp, nbands=self.session.dp.n_sbands,
                            overlap=self.session.dp.sb_overlap, crossval=self.cross_val, nfolds=self.session.dp.n_folds, test_perc=self.session.dp.test_perc)
        setup.process()
        # print(setup.is_sbcsp, self.session.dp.is_sbcsp)
        setup.save_setup(PATH_TO_SESSION + self.session.info.nickname)
        self.acc_val = round(setup.acc * 100, 2)
        self.kpa_val = round(setup.kpa, 4)
        self.update_settings()
        pup = PopupCalResults(self.session, self.acc_cal, self.acc_val, self.kpa_val, self.cross_val)
        popup = Popup(title=":: RESULTADOS DE VALIDAÇÃO OFF-LINE ::", content=pup, size_hint=(None, None), size=(500, 200), title_size=18, title_color=[0.243, 0.663, 0.859, 1], title_align='center')

        popup.open()

    def save_opt_config(self, setup):
        while (setup['tmax'] - setup['tmin']) < 1: setup['tmax'] += 0.5  # garante janela minima de 1seg
        self.session.dp.auto_cal = self.ids.check_auto_cal.active
        self.session.dp.n_iter = self.ids.n_iter.value
        self.session.dp.f_low = int(setup['fl'])
        self.session.dp.f_high = int(setup['fh']) # + int(setup['fl'])
        self.session.dp.epoch_start = setup['tmin']
        self.session.dp.epoch_end = setup['tmax']
        self.session.dp.epoch_len = self.session.dp.eeg_info['fs'] * (setup['tmax'] - setup['tmin'])
        self.session.dp.ncsp = int(setup['ncsp'])
        if setup['approach'] == 1:
            self.session.dp.is_sbcsp = True
            if int(setup['nbands']) > int(setup['fh'] - setup['fl']): setup['nbands'] = int(setup['fh'] - setup['fl']) # - 1
            self.session.dp.n_sbands = int(setup['nbands'])
            self.session.dp.sb_overlap = True
            self.session.dp.sb_clf = {'model':'LDA', 'lda_solver':'svd'}
        else:
            self.session.dp.is_sbcsp = False
            self.session.dp.n_sbands = None
            self.session.dp.sb_overlap = None
            self.session.dp.sb_clf = None

        if setup['filt'] == 0:
            self.session.dp.filt_design = 'DFT'
            self.session.dp.filt_order = None
        if setup['filt'] == 1:
            self.session.dp.filt_design = 'IIR'
            self.session.dp.filt_order = setup['iir_order']
        elif setup['filt'] == 2:
            self.session.dp.filt_design = 'FIR'
            self.session.dp.filt_order = setup['fir_order']

        if setup['clf'] == 0: self.session.dp.final_clf = {'model': 'LR'}
        elif setup['clf'] == 1:
            lda_solver = 'svd' if setup['lda_solver'] == 0 else 'lsqr' if setup['lda_solver'] == 1 else 'eigen'
            # if lda_solver == 'svd': shrink = None
            # else: shrink=None if setup['shrinkage']==0 else 'auto' if setup['shrinkage']==1 else {'shrinkage_float':setup['shrinkage_float']}
            self.session.dp.final_clf = {'model': 'LDA', 'lda_solver': lda_solver} # 'shrinkage':shrink}
        elif setup['clf'] == 2:
            mf = 'euclidean' if setup['metric']==0 else 'manhattan' if setup['metric']==1 else 'minkowski' if setup['metric']==2 else 'chebyshev'
            self.session.dp.final_clf = {'model': 'KNN', 'neig': int(setup['neig']), 'metric': mf} # 'p':setup['p']}
        elif setup['clf'] == 3:
            kf = 'linear' if setup['kernel']==0 else 'poly' if setup['kernel']==1 else 'sigmoid' if setup['kernel']==2 else 'rbf'
            kernel = {'kf': kf} # 'degree': setup['degree']} if setup['kernel'] == 1 else {'kf': kf}
            # gamma = 'scale' if setup['gamma']==0 else 'auto' if setup['gamma']==1 else {'gamma_float':setup['gamma_float']}
            self.session.dp.final_clf = {'model': 'SVM', 'C': setup['C'], 'kernel': kernel, 'gamma': 'scale'}
        elif setup['clf'] == 4:
            af = 'identity' if setup['activ']==0 else 'logistic' if setup['activ']==1 else 'tanh' if setup['activ']==2 else 'relu'
            mlp_solver = 'adam' if setup['mlp_solver'] == 0 else 'lbfgs' if setup['mlp_solver'] == 1 else 'sgd'
            # eta_type = 'constant' if setup['eta_type']==0 else 'invscaling' if setup['eta_type']==1 else 'adaptive'
            self.session.dp.final_clf = {'model': 'MLP', 'eta': setup['eta'], 'activ': {'af': af}, 'mlp_solver': mlp_solver,
                                         'n_neurons': int(setup['n_neurons']), 'n_hidden': int(setup['n_hidden'])} #'eta_type':eta_type,'alpha':setup['alpha'],
        # elif setup['clf'] == 5: self.session.dp.final_clf = {'model': 'Bayes'}
        # elif setup['clf'] == 6:
        #     criterion = 'gini' if setup['crit'] == 0 else 'entropy'
        #     # max_depth = None if setup['max_depth'] == 0 else {'max_depth_int':setup['max_depth_int']}
        #     self.session.dp.final_clf = {'model':'DTree', 'crit':criterion, 'max_depth':None, 'min_split':2} # 'max_depth':max_depth,'min_split':setup['min_split']
        self.session.dp.flag_setup = True
        self.session.saveSession()

    def save_config(self):
        self.session.dp.auto_cal = self.ids.check_auto_cal.active
        self.session.dp.n_iter = self.ids.n_iter.value if self.ids.check_auto_cal.active else None

        self.session.dp.f_low = self.ids.f_low.value
        self.session.dp.f_high = self.ids.f_high.value
        self.session.dp.epoch_start = self.ids.epoch_start.value
        self.session.dp.epoch_end = self.ids.epoch_end.value
        self.session.dp.epoch_len = self.ids.buf_len.value
        self.session.dp.ncsp = self.ids.csp_nei.value
        # self.session.dp.max_amp = self.ids.max_amp.value
        # self.session.dp.max_mse = self.ids.max_mse.value

        self.session.dp.is_sbcsp = self.ids.sb_method.active
        self.session.dp.n_sbands = self.ids.n_sbands.value if self.ids.sb_method.active else None
        self.session.dp.sb_overlap = self.ids.overlap.active if self.ids.sb_method.active else None
        self.session.dp.sb_clf = {'model':'LDA', 'lda_solver':'svd'} if self.ids.sb_method.active else None

        self.session.dp.filt_design = self.ids.filt_approach.text
        self.session.dp.filt_order = self.ids.f_order.value if self.ids.filt_approach.text in ['FIR', 'IIR'] else None

        final_clf_info = {'model': self.ids.clf1.text}
        if self.ids.clf1.text == 'LDA':
            final_clf_info.update({'lda_solver': str(self.ids.lda_solver.text).lower(), 'shrinkage': None})
        if self.ids.clf1.text == 'SVM':
            # kernel = {'kf': str(self.ids.svm_kernel.text).lower(), 'degree': 3} if str(self.ids.svm_kernel.text).lower() == 'poly' else {'kf': str(self.ids.svm_kernel.text).lower()}
            kernel = {'kf': str(self.ids.svm_kernel.text).lower()}
            final_clf_info.update({'C': self.ids.svm_c.value, 'kernel': kernel, 'gamma': 'scale'})
        if self.ids.clf1.text == 'KNN':
            final_clf_info.update({'neig': self.ids.neighbors_knn.value, 'metric': str(self.ids.metric_knn.text).lower()})
            # final_clf_info.update({'neig': self.ids.neighbors_knn.value, 'metric': str(self.ids.metric_knn.text).lower(), 'p': 2}) # with p
        if self.ids.clf1.text == 'MLP':
            final_clf_info.update(
                {'eta': self.ids.mlp_eta.value, # 'alpha': ids.mlp_alpha.value,
                 'activ': {'af': self.ids.mlp_activation.text},
                 'n_neurons': self.ids.mlp_hidden_neurons.value, 'n_hidden': self.ids.mlp_hidden_size.value,
                 'eta_type': 'adaptive',
                 'mlp_solver': str(self.ids.mlp_solver.text).lower()})  # 'eta_type':{'eta_type':'constant'}
        # if self.ids.clf1.text == 'DTree':
        #     final_clf_info.update({'crit': self.ids.dtree_criterion.text, 'max_depth': None, 'min_split': 2})
        self.session.dp.final_clf = final_clf_info

        self.session.dp.flag_setup = True
        self.session.saveSession()

    def update_settings(self):
        # self.ids.max_amp.value = self.session.dp.max_amp
        # self.ids.max_mse.value = self.session.dp.max_mse
        # self.ids.clf2.text = self.session.dp.sb_clf
        self.ids.check_auto_cal.active = self.session.dp.auto_cal
        if self.session.dp.auto_cal:
            self.ids.n_iter.value = self.session.dp.n_iter
            self.ids.load_last_setup.active = True

        self.ids.f_low.value = self.session.dp.f_low
        self.ids.f_high.value = self.session.dp.f_high
        self.ids.epoch_start.value = self.session.dp.epoch_start
        self.ids.epoch_end.value = self.session.dp.epoch_end
        self.ids.buf_len.value = self.session.dp.epoch_len
        self.ids.csp_nei.value = self.session.dp.ncsp

        self.ids.sb_method.active = self.session.dp.is_sbcsp
        if self.session.dp.is_sbcsp:
            self.ids.n_sbands.value = self.session.dp.n_sbands
            self.ids.overlap.active = self.session.dp.sb_overlap

        self.ids.filt_approach.text = self.session.dp.filt_design
        if self.session.dp.filt_design in ['FIR', 'IIR']: self.ids.f_order.value = self.session.dp.filt_order

        self.ids.clf1.text = self.session.dp.final_clf['model']
        if self.ids.clf1.text == 'LDA':
            self.ids.lda_solver.text = str(self.session.dp.final_clf['lda_solver'])
        if self.ids.clf1.text == 'SVM':
            self.ids.svm_c.value = self.session.dp.final_clf['C']
            self.ids.svm_kernel.text = str(self.session.dp.final_clf['kernel']['kf']).capitalize()
            # self.ids.svm_gamma.value = self.session.dp.final_clf['gamma']
        if self.ids.clf1.text == 'KNN':
            self.ids.neighbors_knn.value = self.session.dp.final_clf['neig']
            self.ids.metric_knn.text = str(self.session.dp.final_clf['metric'])
        if self.ids.clf1.text == 'MLP':
            self.ids.mlp_eta.value = self.session.dp.final_clf['eta']
            # self.ids.mlp_alpha.value = self.session.dp.final_clf['alpha']
            self.ids.mlp_hidden_size.value = self.session.dp.final_clf['n_hidden']
            self.ids.mlp_hidden_neurons.value = self.session.dp.final_clf['n_neurons']
            self.ids.mlp_activation.text = self.session.dp.final_clf['activ']['af']
            self.ids.mlp_solver.text = str(self.session.dp.final_clf['mlp_solver'])
        # if self.ids.clf1.text == 'DTree':
        #     self.ids.dtree_criterion.text = self.session.dp.final_clf['crit']

        self.set_enabled_field()

    def set_enabled_clf_params(self, *args):
        try: self.ids.cfg_space.remove_widget(LDAparams)
        except: pass

        try: self.ids.cfg_space.remove_widget(SVMparams)
        except: pass

        try: self.ids.cfg_space.remove_widget(KNNparams)
        except: pass

        try: self.ids.cfg_space.remove_widget(MLPparams)
        except: pass

        if self.ids.clf1.text == 'LDA':
            self.ids.cfg_space.add_widget(LDAparams())
        elif self.ids.clf1.text == 'SVM':
            self.ids.cfg_space.add_widget(SVMparams())
        if self.ids.clf1.text == 'KNN':
            self.ids.cfg_space.add_widget(KNNparams())
        if self.ids.clf1.text == 'MLP':
            self.ids.cfg_space.add_widget(MLPparams())

    def set_enabled_field(self, *args):
        if self.ids.check_auto_cal.active:
            # self.ids.buf_len.disabled = True
            self.ids.item_load_last_setup.disabled = False
            self.ids.load_last_setup.disabled = False
            self.ids.n_iter.disabled = False
            self.ids.epoch_start.disabled = True
            self.ids.epoch_end.disabled = True
            self.ids.sb_method.disabled = True
            self.ids.item_sb_method.disabled = True
            self.ids.n_sbands.disabled = True
            self.ids.overlap.active = True
            self.ids.overlap.disabled = True
            self.ids.item_overlap.disabled = True
            self.ids.filt_approach.disabled = True
            self.ids.item_filt_approach.disabled = True
            self.ids.f_low.disabled = True
            self.ids.f_high.disabled = True
            self.ids.f_order.disabled = True
            self.ids.csp_nei.disabled = True
            self.ids.clf1.disabled = True
            self.ids.item_clf1.disabled = True

            self.ids.svm_c.disabled = True
            self.ids.svm_kernel.disabled = True
            self.ids.item_svm_kernel.disabled = True
            self.ids.metric_knn.disabled = True
            self.ids.neighbors_knn.disabled = True
            self.ids.mlp_eta.disabled = True
            self.ids.mlp_hidden_size.disabled = True
            self.ids.mlp_hidden_neurons.disabled = True
            self.ids.mlp_activation.disabled = True
            self.ids.item_mlp_activation.disabled = True
            self.ids.lda_solver.disabled = True
            self.ids.item_lda_solver.disabled = True


            # self.ids.clf2.disabled = True
            # self.ids.item_clf2.disabled = True
            # self.ids.mlp_alpha.disabled = True
        else:
            self.epoch_lim_max = self.session.dp.eeg_info['trial_tpause']
            self.epoch_lim_min = 0 - self.session.dp.eeg_info['trial_tcue']
            self.srate = self.session.dp.eeg_info['fs']
            # self.ids.clf2.disabled = False
            # self.ids.item_clf2.disabled = False
            # self.ids.mlp_alpha.disabled = False
            # self.ids.buf_len.disabled = False
            self.ids.item_load_last_setup.disabled = True
            self.ids.load_last_setup.disabled = True
            self.ids.n_iter.disabled = True
            self.ids.epoch_start.disabled = False
            self.ids.epoch_end.disabled = False
            self.ids.sb_method.disabled = False
            self.ids.item_sb_method.disabled = False
            self.ids.n_sbands.disabled = False
            self.ids.overlap.disabled = False
            self.ids.item_overlap.disabled = False
            self.ids.filt_approach.disabled = False
            self.ids.item_filt_approach.disabled = False
            self.ids.f_low.disabled = False
            self.ids.f_high.disabled = False
            self.ids.f_order.disabled = False
            self.ids.csp_nei.disabled = False
            self.ids.clf1.disabled = False
            self.ids.item_clf1.disabled = False

            self.ids.svm_c.disabled = False
            self.ids.svm_kernel.disabled = False
            self.ids.item_svm_kernel.disabled = False
            self.ids.neighbors_knn.disabled = False
            self.ids.metric_knn.disabled = False
            self.ids.item_metric_knn.disabled = False
            self.ids.mlp_eta.disabled = False
            self.ids.mlp_hidden_size.disabled = False
            self.ids.mlp_hidden_neurons.disabled = False
            self.ids.mlp_activation.disabled = False
            self.ids.item_mlp_activation.disabled = False
            self.ids.mlp_solver.disabled = False
            self.ids.item_mlp_solver.disabled = False
            self.ids.lda_solver.disabled = False
            self.ids.item_lda_solver.disabled = False

            if self.ids.filt_approach.text not in ['FIR', 'IIR']:
                self.ids.f_order.disabled = True

            if not self.ids.sb_method.active:
                self.ids.n_sbands.disabled = True
                self.ids.overlap.disabled = True

            if self.ids.clf1.text == 'MLP':
                self.ids.svm_c.disabled = True
                self.ids.svm_kernel.disabled = True
                self.ids.item_svm_kernel.disabled = True
                self.ids.metric_knn.disabled = True
                self.ids.item_metric_knn.disabled = True
                self.ids.neighbors_knn.disabled = True
                # self.ids.dtree_criterion.disabled = True
                # self.ids.item_dtree_criterion.disabled = True
                self.ids.lda_solver.disabled = True
                self.ids.item_lda_solver.disabled = True
            elif self.ids.clf1.text == 'SVM':
                # self.ids.mlp_alpha.disabled = True
                self.ids.neighbors_knn.disabled = True
                self.ids.metric_knn.disabled = True
                self.ids.item_metric_knn.disabled = True
                # self.ids.dtree_criterion.disabled = True
                # self.ids.item_dtree_criterion.disabled = True
                self.ids.mlp_eta.disabled = True
                self.ids.mlp_hidden_size.disabled = True
                self.ids.mlp_hidden_neurons.disabled = True
                self.ids.mlp_activation.disabled = True
                self.ids.item_mlp_activation.disabled = True
                self.ids.mlp_solver.disabled = True
                self.ids.item_mlp_solver.disabled = True
                self.ids.lda_solver.disabled = True
                self.ids.item_lda_solver.disabled = True
            elif self.ids.clf1.text == 'KNN':
                # self.ids.mlp_alpha.disabled = True
                self.ids.svm_c.disabled = True
                self.ids.svm_kernel.disabled = True
                self.ids.item_svm_kernel.disabled = True
                # self.ids.dtree_criterion.disabled = True
                # self.ids.item_dtree_criterion.disabled = True
                self.ids.mlp_solver.disabled = True
                self.ids.item_mlp_solver.disabled = True
                self.ids.mlp_eta.disabled = True
                self.ids.mlp_hidden_size.disabled = True
                self.ids.mlp_hidden_neurons.disabled = True
                self.ids.mlp_activation.disabled = True
                self.ids.item_mlp_activation.disabled = True
                self.ids.lda_solver.disabled = True
                self.ids.item_lda_solver.disabled = True
            elif self.ids.clf1.text == 'LDA':
                # self.ids.mlp_alpha.disabled = True
                self.ids.svm_c.disabled = True
                self.ids.svm_kernel.disabled = True
                self.ids.item_svm_kernel.disabled = True
                self.ids.neighbors_knn.disabled = True
                self.ids.metric_knn.disabled = True
                self.ids.item_metric_knn.disabled = True
                # self.ids.dtree_criterion.disabled = True
                # self.ids.item_dtree_criterion.disabled = True
                self.ids.mlp_solver.disabled = True
                self.ids.item_mlp_solver.disabled = True
                self.ids.mlp_eta.disabled = True
                self.ids.mlp_hidden_size.disabled = True
                self.ids.mlp_hidden_neurons.disabled = True
                self.ids.mlp_activation.disabled = True
                self.ids.item_mlp_activation.disabled = True
            else:
                self.ids.svm_c.disabled = True
                self.ids.svm_kernel.disabled = True
                self.ids.item_svm_kernel.disabled = True
                self.ids.neighbors_knn.disabled = True
                self.ids.metric_knn.disabled = True
                self.ids.item_metric_knn.disabled = True
                self.ids.mlp_solver.disabled = True
                self.ids.item_mlp_solver.disabled = True
                self.ids.mlp_eta.disabled = True
                self.ids.mlp_hidden_size.disabled = True
                self.ids.mlp_hidden_neurons.disabled = True
                self.ids.mlp_activation.disabled = True
                self.ids.item_mlp_activation.disabled = True
                self.ids.lda_solver.disabled = True
                self.ids.item_lda_solver.disabled = True


class LDAparams(GridLayout):
    def __init__(self, **kwargs):
        super(LDAparams, self).__init__(**kwargs)

class KNNparams(GridLayout):
    def __init__(self, **kwargs):
        super(KNNparams, self).__init__(**kwargs)

class SVMparams(GridLayout):
    def __init__(self, **kwargs):
        super(SVMparams, self).__init__(**kwargs)

class MLPparams(GridLayout):
    def __init__(self, **kwargs):
        super(MLPparams, self).__init__(**kwargs)

class PopupCalResults(BoxLayout):
    def __init__(self, session, acc_cal, acc_val, kpa_val, cross_val, **kwargs):
        super(PopupCalResults, self).__init__(**kwargs)
        self.session = session
        self.orientation = 'vertical'

        calBox = BoxLayout()
        calBox.add_widget(Label(text='Acurácia [Treino]: ', font_size=22))
        calBox.add_widget(Label(text=str(acc_cal) + '%', font_size=22))

        valBox = BoxLayout()
        if cross_val: valBox.add_widget(Label(text='Acurácia [VCM]: ', font_size=22))
        else: valBox.add_widget(Label(text='Acurácia [Teste]: ', font_size=22))
        valBox.add_widget(Label(text=str(acc_val) + '%', font_size=22))

        self.add_widget(calBox)
        self.add_widget(valBox)

        if not cross_val:
            kpaBox = BoxLayout()
            kpaBox.add_widget(Label(text='Kappa [Teste]: ', font_size=22))
            kpaBox.add_widget(Label(text=str(kpa_val), font_size=22))
            self.add_widget(kpaBox)
