# -*- coding: utf-8 -*-
from processing.utils import PATH_TO_SESSION
from processing.processor import Approach
from view.session_info import UserSession
from processing.sample_manager import SampleManager
import time
import numpy as np

class StartStream():
    def __init__(self):
        self.session = UserSession()
        self.session.info.nickname = 'teste'
        self.session.loadSession()
        
        self.ap = Approach()
        self.ap.loadSetup(PATH_TO_SESSION + self.session.info.nickname)

              
if __name__ == "__main__":
    # run = StartStream()
    # print(run.session.dp.test_data_path)
    
    session = UserSession()
    session.info.nickname = 'teste'
    session.loadSession()

    sm = SampleManager(session.acq.com_port, session.dp.epoch_start, session.dp.epoch_end,
                       int(session.dp.buf_len), session.dp.class_ids, daisy=False,
                       mode='simu', path=session.acq.path_to_file,
                       labels_path=session.acq.path_to_labels_file, dummy=False)
    
    sm.daemon = True
    sm.stop_flag = False
    sm.start()
    
    time.sleep(2)
    
    sm.Stop()
    
    # print(int(session.dp.buf_len))
    
    # print(sm.current_playback_label)
    
    # print(sm.board.playback_data.shape)
    
    
    # ap = Approach()
    # ap.loadSetup(PATH_TO_SESSION + session.info.nickname)

    # setup_acc = ap.learner.get_results()
    
    # print(setup_acc)
    
    E = np.load('/home/vboas/cloud/devto/overmind/userdata/teste1/acqs/events_03-02-2020_01h19m.npy')
    