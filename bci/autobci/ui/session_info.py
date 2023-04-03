# coding: utf-8
import pickle # manipular dados binarios
from proc.utils import PATH_TO_SESSION

class UserSession:
    def __init__(self):
        self.info = InfoHeader()            # GENERAL SETTINGS
        self.acq = AcquisitionHeader()      # ACQUISITION SETTINGS
        self.dp = DataProcessingHeader()    # DATA PROCESSING SETTINGS
        self.precal = PreCalHeader()        # PRE-CALIBRATION SETTINGS
        self.control = ControlHeader()      # CONTROL SETTINGS
        self.path = PATH_TO_SESSION

    def saveSession(self):
        # path = PATH_TO_SESSION + self.info.nickname + '/' + 'session_info.pkl' # escrita binaria wb (write byte)
        path = self.path + self.info.nickname + '/' + 'session_info.pkl' # escrita binaria wb (write byte)
        with open(path, 'wb') as file_name: pickle.dump(self.__dict__, file_name) # salva o dicionário da sessao em arquivo binario .pickle

    def loadSession(self):
        # print(self.info.nickname)
        # path = PATH_TO_SESSION + self.info.nickname + '/' + 'session_info.pkl' # leitura binaria 'rb' (read byte)
        path = self.path + self.info.nickname + '/' + 'session_info.pkl' # leitura binaria 'rb' (read byte)
        with open(path, 'rb') as file_name: load_obj = pickle.load(file_name) # carrega em load_obj os dados binarios
        self.__dict__.update(load_obj) # atualiza todos os atributos da sessÃ£o carregada com os dados em load_obj

class InfoHeader:
    def __init__(self):
        self.flag = False # controla usuÃ¡rio logado ou nao
        self.is_dataset = False # diz se o usuario cadastrado pertence a um dataset publico ou nao
        self.nickname = None
        self.fullname = None
        self.save_date = None
        self.age = None
        self.gender = None
        self.pnd = False
        self.ds_name = None # usado somente para usuarios de data sets publicos (is_dataset == True)
        self.ds_subject = None # usado somente para usuarios de data sets publicos (is_dataset == True)

class AcquisitionHeader:
    def __init__(self):
        self.flag_mode = False # True ao salvar AcqMode
        self.flag_protocol = False
        self.mode = None
        self.board = 'OpenBCI Cyton'
        self.com_port = None
        self.daisy = None
        self.ch_labels = None
        self.sample_rate = None
        self.eeg_path_simu_acq = None  # simulator
        self.class_ids = None # simulator
        self.dummy = None  # simulator
        self.n_runs = None
        self.n_trials = None
        self.cue_offset = None
        self.cue_time = None
        self.min_pause = None
        self.trial_duration = None
        self.runs_interval = None

class DataProcessingHeader:
    def __init__(self):
        self.flag_load = False  # True ao salvar CalLoad
        self.flag_setup = False

        self.eeg_path_train = None
        self.eeg_path_test = None
        self.eeg_info = None
        self.class_ids = None
        self.cross_val = False
        self.n_folds = None
        self.test_perc = None

        self.auto_cal = False
        self.n_iter = None
        self.f_low = None
        self.f_high = None
        self.epoch_start = None
        self.epoch_end = None
        self.epoch_len = None
        self.ncsp = None
        self.sb_clf = None
        self.sb_overlap = True

        self.channels = None
        self.final_clf = None
        self.filt_design = None
        self.filt_order = None  # if filt_approach == 'IIR':
        self.is_sbcsp = False
        self.n_sbands = None # if sb_method:

        # self.max_amp = None
        # self.max_mse = None
        # self.ch_labels = None
        # self.sample_rate = None
        # self.new_path = False
        # self.train_events_path = None
        # self.test_data_path = None
        # self.test_events_path = None
        # self.max_channels = None
        # self.trials_per_class = None
        # self.trial_tcue = None
        # self.trial_tpause = None
        # self.trial_mi_time = None

class ControlHeader:
    def __init__(self):
        self.flag = False
        self.game_threshold = None
        self.window_overlap = None
        self.warning_threshold = None
        self.forward_speed = None
        self.inst_prob = None
        self.keyb_enable = None
        self.action_cmd1 = None
        self.action_cmd2 = None

        self.space_tta = 2
        self.space_delta_t = 0.2
        self.space_threshold = 80
        self.space_overlap = 0.5  # 0 a 1 - controla qto do ultimo check influenciará o próximo check de comando (funcão de tta)
        self.space_arrasto = 0.5  # 0.65
        self.space_delta_check = 1.3  # (1 - self.arrasto) * (self.tta - (self.tta * self.overlap)) # 0.21
        
class PreCalHeader:
    def __init__(self):
        self.flag = False
        self.ch_energy_left = None
        self.ch_energy_right = None
        self.total_time = None
        self.relax_time = None
        self.sign_direction = None
        self.plot_flag = None
