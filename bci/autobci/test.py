# coding: utf-8
import os
import time
import pickle
import collections
from kivy.clock import Clock
from proc.utils import PATH_TO_SESSION
from ui.session_info import UserSession
from proc.sample_manager import SampleManager
# from autobci.proc.sample_manager import SampleManager

# L = tamanho do buffer (smax - smin)
# entrego ao simulador de streaming, duas epocas de largura L sorteadas entre as epocas da classe de indice [winning = 0 ou 1]
# inicio o streaming
# a cada 0.004s:
# 	add [1,ch] em circBuff(max=L) ## simula a entrada pelo amplificador (armazena as últimas Lxch amostras)
# 	add 1 int em tBuff(max=L)
# 	add [1,ch] no conteiner storeData ## é um storage permanente (mantém todo o streaming para armazenamento no disco)
# 	# quando cheio os buffers vao deslizando (excluem a amostra mais antiga para add uma nova amostra)
# quando a fonte do streaming simulado estiver acabando (faltando 30)... add mais uma época

class Teste():
    def __init__(self):
        self.session = UserSession()
        self.session.info.nickname = 'dfsdsf'
        path = self.session.path + self.session.info.nickname + '/'
        if not os.path.isdir(path): 
            os.makedirs(path)
            self.session.saveSession()
        else: self.session.loadSession()
        
    def start(self):
        U1_local = collections.deque(maxlen=int(50)) # U1_local,U2_local registram acumuladores deslizantes (instantaneo)
        U2_local = collections.deque(maxlen=int(50)) # das classes A,B respectivamente (usados p verificar envio comando)
        self.sm = SampleManager(self.session.acq.sample_rate, None, buf_len=int(self.session.dp.epoch_len),
                                tmin=self.session.dp.epoch_start, tmax=self.session.dp.epoch_end, 
                                class_ids=self.session.acq.class_ids,
                                mode=self.session.acq.mode, path=self.session.acq.eeg_path_simu_acq)
        self.sm.daemon = True
        self.sm.stop_flag = False
        self.sm.start()
        Clock.schedule_interval(self.get_probs, 0.05)
        
    def stop(self):
        self.sm.stop_flag = True
        self.sm.join()
        Clock.unschedule(self.get_probs)
        
    def get_probs(self, dt):
        t, buf = self.sm.GetBuffData() # tira a foto dos buffers atuais
        print(buf.shape[0], self.session.dp.epoch_len)
        # if buf.shape[0] == session.dp.epoch_len: p = ap.classify_epoch(buf.T, 'prob')[0]
        

if __name__ == "__main__":
    obj = pickle.load(open(os.path.dirname(__file__) + '/data/vitor/session_info.pkl', 'rb'))
    
    session = UserSession()
    session.info.nickname = 'vitor'
    session.loadSession()
    # print('eeg_path:', session.acq.eeg_path_simu_acq)
    # print('flag =', session.control.flag)
    # print('window_overlap =', session.control.window_overlap) 
    # print('game_threshold =', session.control.game_threshold)
    # print('warning_threshold =', session.control.warning_threshold)
    # print('inst_prob =', session.control.inst_prob)
    # print('forward_speed =', session.control.forward_speed) 
    # print('keyb_enable =', session.control.keyb_enable)
    # print('action_cmd1 =', session.control.action_cmd1) 
    # print('action_cmd2 =', session.control.action_cmd2)
    # session.info.nickname = 'vitor'
    # session.saveSession()
    
    # teste = Teste()
    # teste.start()
    # time.sleep(30)
    # teste.stop()
    
    # import collections
    # import math
    # ABUF_LEN = math.ceil(35.76970443349754)
    # U1_local = collections.deque(maxlen=ABUF_LEN)  
    
    # A = [1,2,3,4,5,6,7,8,9]
    # A = iter(A)
    # next(A)