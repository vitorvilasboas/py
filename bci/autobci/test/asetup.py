# -*- coding: utf-8 -*-
import math
import pickle
import itertools
import numpy as np
import scipy.signal as sp
import scipy.linalg as lg
import scipy.stats as sst
from sklearn.svm import SVC
from scipy.fftpack import fft
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from mne import Epochs, pick_types, find_events
# from mne.decoding import CSP #Import Common Spatial Patterns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report, make_scorer, accuracy_score, precision_recall_fscore_support, \
    confusion_matrix, cohen_kappa_score
from proc.processor import Filter, Processor
from proc.utils import PATH_TO_SESSION, load_npy_data, readEvents, extractEpochs, nanCleaner, load_pickle_data

class Approach:
    def __init__(self, session=None):
        self.session = session

    def define_approach(self, sb_approach, sample_rate, f_low, f_high, csp_nei, class_ids, epoch_start, epoch_end,
                        filt_type, clf_sb, clf_final, order=None, nbands=None, overlap=True, crossval=True, nfolds=10,
                        test_perc=0.2):
        self.class_ids = class_ids
        self.sample_rate = sample_rate
        self.f_low = f_low
        self.f_high = f_high
        self.smin = int(math.floor(epoch_start * sample_rate))
        self.smax = int(math.floor(epoch_end * sample_rate))
        self.buffer_len = self.smax - self.smin
        self.crossval = crossval
        self.nfolds = nfolds
        self.test_perc = test_perc
        self.ncsp = csp_nei

        self.dft_rf = self.sample_rate / self.buffer_len  # resolução em frequência fft
        self.dft_size_band = round(2 / self.dft_rf)  # 2 representa sen e cos que foram separados do componente complexo da fft intercalados

        self.learner = Processor()
        self.learner.design_CLF(clf_final)
        self.learner.design_CSP(self.ncsp)

        self.sb_approach = sb_approach
        if self.sb_approach:
            if nbands > (self.f_high - self.f_low): nbands = (self.f_high - self.f_low)
            self.nbands = nbands
            self.overlap = overlap
            self.learner.design_CLF(clf_sb, sb_level=True)

        self.filt_type = filt_type
        if self.filt_type in ['IIR','FIR']: self.order = order

        self.filter = Filter(f_low, f_high, self.buffer_len, sample_rate, forder=order, filt_type=filt_type)
        # self.learner.assemble_learner()

    def set_channels(self, channels):
        self.channels = channels
        # if self.channels == list([-1]): self.channels = list(np.arange(0, self.session.dp.data_max_channels))

    def set_eeg_path(self, eeg_path):
        self.eeg_path = eeg_path
        self.data, self.events, self.info = self.load_eeg_data(self.eeg_path)

    def load_eeg_data(self, eeg_path, data_format='path'):
        if data_format == 'path':
            data, events, info = np.load(eeg_path, allow_pickle=True) # load_pickle_data(eeg_path)
            data = data[self.channels]
            events = events.astype(int)
            # if self.channels == [-1]: data = data[:self.session.dp.max_channels]
            # else: data = data[self.channels]
        elif data_format == 'npy':
            data, events, info = eeg_path
        return data, events, info

    def validate_model(self):
        # data, events = self.load_data(self.data_cal_path, self.events_cal_path)
        data, events, info = self.data, self.events, self.info # self.load_eeg_data(self.eeg_path)
        epochs, labels = extractEpochs(data, events, self.smin, self.smax, self.class_ids)
        epochs = nanCleaner(epochs)
        self.learner.evaluate(self, epochs, labels)
        score = self.learner.get_results()
        return score

    def classify_set(self, epochs, labels):
        self.learner.evaluate_set(epochs, labels)
        score = self.learner.get_results()
        return score

    def classify_epoch(self, epoca, out_param='label'):
        # epoch_f = nanCleaner(epoca)
        epoch_f = epoca
        # epoch_f = self.preprocess(epoch)
        if not epoca == []:
            guess = self.learner.evaluate_epoch(self, epoch_f, out_param=out_param)
        else: guess = None
        return guess

    def set_cal_path(self, dpath):
        self.data_path = dpath

    def load_epochs(self, data, events):
        epochs, labels = extractEpochs(data, events, self.smin, self.smax, self.class_ids)
        return epochs, labels

    def preprocess(self, data_in):
        data = nanCleaner(data_in)
        data_out = self.filter.apply_filter(data)
        return data_out

    def saveSetup(self, path):
        self.data = self.events = self.info = None
        pickle.dump(self.__dict__, open(path + '/setup_info.pkl', 'wb'))

    def loadSetup(self, path):
        self.__dict__.update(pickle.load(open(path + '/setup_info.pkl', 'rb')))

    # def set_cal_path_old(self, dpath, evpath):
    #     self.data_cal_path = dpath
    #     self.events_cal_path = evpath

    # def set_val_path(self, dpath, evpath):
    #     self.data_val_path = dpath
    #     self.events_val_path = evpath
    #     self.data, self.events = self.load_data(self.data_val_path, self.events_val_path) ## add
    #     self.epochs, self.labels = extractEpochs(self.data, self.events, self.smin, self.smax, self.class_ids)
    #     self.epochs = nanCleaner(self.epochs)

    # def load_data(self, dpath, evpath, data_format='path'):
    #     if data_format == 'path':
    #         # if self.channels == [-1]:
    #         #     data = load_npy_data(dpath)
    #         #     max_channels = data.shape[0]
    #         #     data = data[:self.session.acq.ds_max_channels]
    #         # else:
    #         #     data = load_npy_data(dpath)[self.channels]
    #         data = load_npy_data(dpath)[self.channels]
    #         events = readEvents(evpath)
    #     elif data_format == 'npy':
    #         data, events = dpath, evpath
    #     return data, events

