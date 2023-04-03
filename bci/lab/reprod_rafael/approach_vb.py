# -*- coding: utf-8 -*-
import math
import pickle
from processor_vb import Learner, Filter
from utils_vb import nanCleaner
from handler_vb import load_data_as_np, read_events, extract_epochs

class Approach:
    def __init__(self):
        pass

    def define_approach(self, sample_rate, f_low, f_high, f_order, csp_nei, class_ids, epoch_start, epoch_end):
        self.class_ids = class_ids
        # FEATURE EXTRACTION:
        #self.smin = int(math.floor(int(epoch_start) * int(sample_rate)))
        self.smin = int(math.floor(epoch_start * sample_rate))
        self.smax = int(math.floor(epoch_end * sample_rate))
        self.filter = Filter(f_low, f_high, sample_rate, f_order, filt_type='iir', band_type='band')
        self.learner = Learner()
        self.learner.design_LDA()
        self.learner.design_CSP(csp_nei)
        self.learner.assemble_learner()

    def train_model(self):
        data, ev = self.load_data(self.data_cal_path, self.events_cal_path)
        epochs, labels = self.load_epochs(data, ev)
        epochs = self.preprocess(epochs)
        self.learner.learn(epochs, labels)
        self.learner.evaluate_set(epochs, labels)
        score = self.learner.get_results()
        return score

    def validate_model(self):
        data, ev = self.load_data(self.data_val_path, self.events_val_path)
        epochs, labels = self.load_epochs(data, ev)
        epochs = self.preprocess(epochs)
        self.learner.evaluate_set(epochs, labels)
        score = self.learner.get_results()
        return score

    def cross_validate_model(self, n_iter, test_perc):
        data, ev = self.load_data(self.data_cal_path, self.events_cal_path)
        epochs, labels = self.load_epochs(data, ev)
        epochs = self.preprocess(epochs)
        score = self.learner.cross_evaluate_set(epochs, labels, n_iter, test_perc)
        return score

    def classify_set(self, epochs, labels):
        self.learner.evaluate_set(epochs, labels)
        score = self.learner.get_results()
        return score

    def classify_epoch(self, epoch, out_param='label'):
        epoch_f = self.preprocess(epoch)
        if not epoch == []: guess = self.learner.evaluate_epoch(epoch_f, out_param=out_param)
        else: guess = None
        return guess

    def set_cal_path(self, dpath, evpath):
        self.data_cal_path = dpath
        self.events_cal_path = evpath

    def set_val_path(self, dpath, evpath):
        self.data_val_path = dpath
        self.events_val_path = evpath

    def load_data(self, dpath, evpath, data_format='path'):
        if data_format == 'path':
            if self.channels == [-1]: 
                data = load_data_as_np(dpath) #load_data_as_np(dpath).T
            else: # esperado
                data = load_data_as_np(dpath).T[self.channels]
            events = read_events(evpath)
        elif data_format == 'npy':
            data, events = dpath, evpath
        return data, events

    def load_epochs(self, data, events):
        epochs, labels = extract_epochs(data, events, self.smin, self.smax, self.class_ids)
        return epochs, labels

    def preprocess(self, data_in):
        data = nanCleaner(data_in)
        data_out = self.filter.apply_filter(data)
        return data_out

    def set_valid_channels(self, channels):
        self.channels = channels

    def save_to_pkl(self, path):
        path += '/approach_info.pkl'
        with open(path, 'w') as file_name: pickle.dump(self.__dict__, file_name)

    def load_pkl(self, path):
        path += '/approach_info.pkl'
        with open(path, 'r') as file_name: load_obj = pickle.load(file_name)
        self.__dict__.update(load_obj)