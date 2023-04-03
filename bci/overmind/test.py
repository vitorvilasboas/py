# coding: utf-8
import pickle
from view.session_info import UserSession

obj = pickle.load(open('/home/vboas/cloud/devto/BCI/overmind/userdata/A1/session_info.pkl', 'rb'))

# session = UserSession()
# session.info.nickname = 'A1'
# session.loadSession()
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