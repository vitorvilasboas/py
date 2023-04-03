import mne
import pickle
import warnings
import itertools
import numpy as np
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.linalg import eigh
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# from mne import Epochs, pick_types, find_events
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import lfilter, butter, firwin, filtfilt, iirfilter, decimate, welch
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, make_scorer, precision_recall_fscore_support, confusion_matrix

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)


def extractEpochs(data, events, smin, smax, class_ids):
    events_list = events[:, 1] # get class labels column
    cond = False
    for i in range(len(class_ids)): cond += (events_list == class_ids[i]) #get only class_ids pos in events_list
    idx = np.where(cond)[0]
    s0 = events[idx, 0] # get initial timestamps of each class epochs
    sBegin = s0 + smin
    sEnd = s0 + smax
    n_epochs = len(sBegin)
    n_channels = data.shape[0]
    n_samples = smax - smin
    epochs = np.zeros([n_epochs, n_channels, n_samples])
    labels = events_list[idx]
    bad_epoch_list = []
    for i in range(n_epochs):
        epoch = data[:, sBegin[i]:sEnd[i]]
        if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch # Check if epoch is complete
        else:
            print('Incomplete epoch detected...', n_samples, '!=', epoch.shape[1])
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    return epochs, labels
    
    
def nanCleaner(epoch):
    """Removes NaN from data by interpolation
    data_in : input data - np matrix channels x samples
    data_out : clean dataset with no NaN samples"""
    for i in range(epoch.shape[0]):
        bad_idx = np.isnan(epoch[i, :])
        epoch[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoch[i, ~bad_idx])
    return epoch
    
    
def corrigeNaN(data):
    for ch in range(data.shape[0] - 1):
        this_chan = data[ch]
        data[ch] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(data[ch])
        meanChannel = np.nanmean(data[ch])
        data[ch, mask] = meanChannel
    return data


class CSP:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None
    def fit(self, X, t):
        ch = X.shape[1]
        class_ids = np.unique(t)
        X1 = X[class_ids[0] == t]
        X2 = X[class_ids[1] == t]
        S1, S2 = np.zeros((ch, ch)), np.zeros((ch, ch))
        for i in range(len(X1)): S1 += np.dot(X1[i], X1[i].T) / X1[i].shape[-1] # cov X[i]
        for i in range(len(X2)): S2 += np.dot(X2[i], X2[i].T) / X2[i].shape[-1] # ...sum((X*X.T)/q)
        S1 /= len(X1);
        S2 /= len(X2)
        [D, W] = eigh(S1, S1 + S2) # + 1e-10 * np.eye(22))
        ind = np.empty(ch, dtype=int)
        ind[0::2] = np.arange(ch-1, ch//2 - 1, -1)
        ind[1::2] = np.arange(0, ch//2)
        # W += 1e-1 * np.eye(22)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        return self # used on cross-validation pipeline
    def transform(self, X):
        Y = np.asarray([np.dot(self.filters_, ep) for ep in X])
        # FEAT = np.log(np.mean(Y**2, axis=2))
        FEAT = np.log(np.var(Y, axis=2))
        return FEAT


class Filter:
    def __init__(self, fl, fh, Fs, ftype='IIR', forder=5, band_type='bandpass'):
        self.ftype = ftype
        if self.ftype != 'DFT':
            nyq = 0.5*Fs
            low = fl/nyq
            high = fh/nyq
            if low == 0: low = 0.001
            if high >= 1: high = 0.99
            if self.ftype == 'IIR':
                self.b, self.a = butter(forder, [low, high], btype=band_type) # iirfilter(..., btype='band')
            elif self.ftype == 'FIR':
                self.b, self.a = firwin(forder, [low, high], window='hamming', pass_zero=False), [1]

    def apply_filter(self, X, is_epoch=False):
        if self.ftype != 'DFT': XF = lfilter(self.b, self.a, X) # lfilter, filtfilt
        else:
            XF = fft(X)
            if is_epoch:
                real, imag = np.real(XF).T, np.imag(XF).T
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))))
            else:
                real = np.transpose(np.real(XF), (2, 0, 1))
                imag = np.transpose(np.imag(XF), (2, 0, 1))
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))), (1, 2, 0))
        return XF


class Processor: # CPS - Central/Chain Processing System
    def __init__(self, session=None):
        self.session = session
        self.channels = None
        self.data_train, self.events_train, self.info_train = None, None, None
        self.data_test, self.events_test, self.info_test = None, None, None
        self.Fs = 250
        self.class_ids = [1, 2]
        self.f_low, self.f_high = 8, 30
        self.filt_type, self.filt_order = 'IIR', 5
        self.tmin, self.tmax = 0.5, 2.5
        self.ncsp, self.ncsp_list = 8, None
        self.nbands, self.sub_bands = None, None
        self.overlap = True
        self.is_sbcsp = False
        self.crossval = False
        self.nfolds, self.test_perc = 10, 0.2
        self.clf_dict = {'model':'LDA', 'lda_solver':'svd'}
        self.filter, self.csp = None, None # self.csp_filters = None
        self.lda, self.norm_a, self.norm_b = None, None, None
        self.clf, self.t, self.y, self.y_prob = None, None, None, None
        self.acc, self.kpa = None, None
        
        self.H = None
        self.acc_cal = None
        self.acc_best = None
        self.learner_cla = None
        self.learner_sb = None
        self.learner_iir = None
        self.as_trials = None
        self.cost = 0

    def define_params(self, f_low=8, f_high=30, ncsp=8, tmin=0.5, tmax=2.5, fs=250., class_ids=[1,2], filt_type='IIR', filt_order=5, is_sbcsp=False,
                      clf_dict={'model':'LDA', 'lda_solver':'svd'}, nbands=None, overlap=True, crossval=True, nfolds=10, test_perc=0.2, ncsp_list=None):
        self.class_ids = class_ids
        self.Fs = fs
        self.f_low = f_low
        self.f_high = f_high
        self.tmin = tmin
        self.tmax = tmax
        self.crossval = crossval
        self.nfolds = nfolds
        self.test_perc = test_perc
        self.ncsp = ncsp
        self.ncsp_list = ncsp_list
        self.is_sbcsp = is_sbcsp
        self.nbands = nbands
        self.overlap = overlap
        self.filt_type = filt_type
        self.filt_order = filt_order
        self.clf_dict = clf_dict

    def save_setup(self, path):
        self.data_train, self.events_train, self.info_train = None, None, None
        self.data_test, self.events_test, self.info_test = None, None, None
        pickle.dump(self.__dict__, open(path + '.pkl', 'wb'))

    def load_setup(self, path):
        self.__dict__.update(pickle.load(open(path + '.pkl', 'rb')))
        
    def clear_eeg_data(self):
        self.data_train, self.events_train, self.info_train = None, None, None
        self.data_test, self.events_test, self.info_test = None, None, None

    def load_eeg_train(self, eeg_path, channels=None, data_format='path'):
        data, events, self.info_train = np.load(eeg_path, allow_pickle=True) # load_pickle_data(eeg_path)
        self.channels = range(0, data.shape[0]) if channels is None else channels
        self.data_train = data[self.channels]
        self.events_train = events.astype(int)
        self.Fs = self.info_train['fs']

    def load_eeg_test(self, eeg_path, channels=None, data_format='path'):
        data, events, self.info_test = np.load(eeg_path, allow_pickle=True) # load_pickle_data(eeg_path)
        self.channels = range(0, data.shape[0]) if channels is None else channels
        self.data_test = data[self.channels]
        self.events_test = events.astype(int)
        
    def process(self, single_eeg=False):
        smin, smax = int(self.tmin * self.Fs), int(self.tmax * self.Fs)
        if single_eeg:
            Z, t = extractEpochs(self.data_train, self.events_train, smin, smax, self.class_ids)
            kf = StratifiedShuffleSplit(1, test_size=0.5, random_state=42)
            for train, test in kf.split(Z, t): ZT, tt, ZV, tv = Z[train], t[train], Z[test], t[test]
        else:
            ZT, tt = extractEpochs(self.data_train, self.events_train, smin, smax, self.class_ids)
            ZV, tv = extractEpochs(self.data_test, self.events_test, smin, smax, self.class_ids)
            Z, t = np.vstack([ZT, ZV]), np.hstack([tt, tv])
        Z, ZT, ZV = nanCleaner(Z), nanCleaner(ZT), nanCleaner(ZV)
        if self.crossval:
            kf = StratifiedShuffleSplit(self.nfolds, test_size=self.test_perc, random_state=42)  # StratifiedKFold(self.nfolds, False)
            cross_acc, cross_kpa = [], []
            for train, test in kf.split(Z, t):
                ZT, tt, ZV, tv = Z[train], t[train], Z[test], t[test]
                acc_fold, kpa_fold = self.sbcsp_chain(ZT, ZV, tt, tv) if self.is_sbcsp else self.classic_chain(ZT, ZV, tt, tv)
                cross_acc.append(acc_fold)
                cross_kpa.append(kpa_fold)
            self.acc, self.kpa = np.mean(cross_acc), np.mean(cross_kpa)
        else: self.acc, self.kpa = self.sbcsp_chain(ZT, ZV, tt, tv) if self.is_sbcsp else self.classic_chain(ZT, ZV, tt, tv)

    def design_clf(self):
        # if self.clf_dict['model'] == 'LDA': self.clf = LDA(solver=self.clf_dict['lda_solver'], shrinkage=None)
        if self.clf_dict['model'] == 'LDA': self.clf = LDA()
        if self.clf_dict['model'] == 'LR':  self.clf = LogisticRegression(verbose=False)
        if self.clf_dict['model'] == 'SVM':
            self.clf = SVC(kernel=self.clf_dict['kernel']['kf'], C=10 ** (self.clf_dict['C']), gamma='scale', degree=3, probability=True)
        if self.clf_dict['model'] == 'KNN':
            self.clf = KNeighborsClassifier(n_neighbors=int(self.clf_dict['neig'])) # , metric=self.clf_dict['metric'], p=3
        if self.clf_dict['model'] == 'MLP':
            self.clf = MLPClassifier(verbose=False, max_iter=10000, learning_rate_init=10 ** self.clf_dict['eta'],
                                     activation=self.clf_dict['activ']['af'], hidden_layer_sizes=(int(self.clf_dict['n_neurons']), int(self.clf_dict['n_hidden'])))
                                     # learning_rate='constant', solver=self.clf_dict['mlp_solver']) # alpha=10 ** clf_dict['alpha'], learning_rate=clf_dict['eta_type'],
            self.clf.out_activation = 'softmax'                    
        if self.clf_dict['model'] == 'Bayes': self.clf = GaussianNB()
        if self.clf_dict['model'] == 'DTree': self.clf = DecisionTreeClassifier(criterion=self.clf_dict['crit'], random_state=0, max_depth=None,  min_samples_split=2)

    def classic_chain(self, ZT, ZV, tt, tv):
        self.filter = Filter(self.f_low, self.f_high, self.Fs, ftype=self.filt_type, forder=self.filt_order)
        XT = self.filter.apply_filter(ZT)
        XV = self.filter.apply_filter(ZV)
        if self.filt_type == 'DFT':  # extrai somente os bins referentes à banda de interesse
            bsize = 2/(self.Fs/ZT.shape[-1])  # 2==sen/cos do comp complexo da fft intercalados / resol freq 
            XT = XT[:, :, round(self.f_low * bsize):round(self.f_high * bsize)]
            bsize = 2/(self.Fs/ZV.shape[-1])  # 2==sen/cos complexo fft intercalados / resol freq
            XV = XV[:, :, round(self.f_low * bsize):round(self.f_high * bsize)]
        self.csp = mne.decoding.CSP(n_components=int(self.ncsp))  # mne.decoding.CSP(...) # CSP_v2(n_components=int(self.ncsp), reg=None, log=True, cov_est='epoch')
        self.csp.fit(XT, tt)
        self.csp_filters = self.csp.filters_
        FT, FV = self.csp.transform(XT), self.csp.transform(XV)
        self.design_clf()
        self.clf.fit(FT, tt)
        self.y, self.t = self.clf.predict(FV), tv
        self.y_prob = self.clf.predict_proba(FV)
        acc, kpa = np.mean(self.y == self.t), cohen_kappa_score(self.y, self.t)
        return acc, kpa

    def sbcsp_chain(self, ZT, ZV, tt, tv):
        if self.nbands > (self.f_high-self.f_low): self.nbands = int(self.f_high-self.f_low)

        if self.overlap:
            step = (self.f_high-self.f_low)/(self.nbands+1) # n_bins/nbands+1
            size = step/0.5  # step/overlap
        else:
            step = (self.f_high-self.f_low)/self.nbands # n_bins/nbands
            size = step

        self.sub_bands = []
        for i in range(self.nbands):
            fl_sb = i * step + self.f_low
            fh_sb = i * step + size + self.f_low
            self.sub_bands.append([fl_sb, fh_sb])

        if self.filt_type == 'DFT':
            self.filter = Filter(self.f_low, self.f_high, self.Fs, ftype=self.filt_type, forder=self.filt_order)
            XTF, XVF = self.filter.apply_filter(ZT), self.filter.apply_filter(ZV)
            bsize = 2/(self.Fs/ZT.shape[-1])  # 2==sen/cos complexo fft intercalados / resol freq
            XT = [XTF[:,:,round(self.sub_bands[i][0]*bsize):round(self.sub_bands[i][1]*bsize)] for i in range(self.nbands)]
            bsize = 2/(self.Fs/ZV.shape[-1])  # 2==sen/cos complexo fft intercalados / resol freq
            XV = [XVF[:,:,round(self.sub_bands[i][0]*bsize):round(self.sub_bands[i][1]*bsize)] for i in range(self.nbands)]
        elif self.filt_type in ['IIR','FIR']:
            self.filter = [Filter(self.sub_bands[i][0], self.sub_bands[i][1], self.Fs, ftype=self.filt_type, forder=self.filt_order) for i in range(self.nbands)]
            XT = [self.filter[i].apply_filter(ZT) for i in range(self.nbands)]
            XV = [self.filter[i].apply_filter(ZV) for i in range(self.nbands)]

        if self.ncsp_list is None: self.csp = [mne.decoding.CSP(n_components=int(self.ncsp), reg=None) for i in range(self.nbands)] # mne.decoding.CSP()
        else: self.csp = [mne.decoding.CSP(n_components=int(self.ncsp_list[i]), reg=None) for i in range(self.nbands)] # to tuning ncsp

        for i in range(self.nbands): self.csp[i].fit(XT[i], tt)
        FT = [self.csp[i].transform(XT[i]) for i in range(self.nbands)]
        FV = [self.csp[i].transform(XV[i]) for i in range(self.nbands)]

        if self.ncsp_list is None: self.csp_filters = [self.csp[i].filters_[:int(self.ncsp)] for i in range(self.nbands)]
        else: self.csp_filters = [self.csp[i].filters_[:int(self.ncsp_list[i])] for i in range(self.nbands)]

        self.lda = [LDA() for i in range(self.nbands)]
        for i in range(self.nbands): self.lda[i].fit(FT[i], tt)
        ST = np.asarray([np.ravel(self.lda[i].transform(FT[i])) for i in range(self.nbands)]).T # Score LDA
        SV = np.asarray([np.ravel(self.lda[i].transform(FV[i])) for i in range(self.nbands)]).T

        ST0 = ST[tt == self.class_ids[0], :]
        ST1 = ST[tt == self.class_ids[1], :]
        self.norm_a = norm(np.mean(ST0, axis=0), np.std(ST0, axis=0))
        self.norm_b = norm(np.mean(ST1, axis=0), np.std(ST1, axis=0))
        META_ST = np.log(self.norm_a.pdf(ST) / self.norm_b.pdf(ST))
        META_SV = np.log(self.norm_a.pdf(SV) / self.norm_b.pdf(SV))

        self.design_clf()
        self.clf.fit(META_ST, tt)
        self.y, self.t = self.clf.predict(META_SV), tv
        self.y_prob = self.clf.predict_proba(META_SV)
        acc, kpa = np.mean(self.y == self.t), cohen_kappa_score(self.y, self.t)
        return acc, kpa

    def classify_epoch(self, epoch, out_param='label'):
        if epoch == []: return None
        else:
            # print(epoch.shape)
            bsize = 2 / (self.Fs / epoch.shape[-1])
            if not self.is_sbcsp:
                X = self.filter.apply_filter(epoch, is_epoch=True)
                if self.filt_type == 'DFT':  # extrai somente os bins referentes à banda de interesse
                    X = X[:, round(self.f_low * bsize):round(self.f_high * bsize)]
                # f = np.log(np.mean(np.dot(self.csp_filters, X)**2, axis=-1))
                f = np.log(np.var(np.dot(self.csp.filters_[:self.ncsp], X), axis=-1))
                y_prob = self.clf.predict_proba(f.reshape(1, -1))
                y_label = self.clf.predict(f.reshape(1, -1))
            else:
                if self.filt_type == 'DFT':
                        XF = self.filter.apply_filter(epoch, is_epoch=True)
                        X = [XF[:, round(self.sub_bands[i][0]*bsize):round(self.sub_bands[i][1]*bsize)] for i in range(self.nbands)]
                elif self.filt_type in ['IIR','FIR']:
                    X = [self.filter[i].apply_filter(epoch) for i in range(self.nbands)]
                # F = [self.csp[i].transform(X[i]) for i in range(self.nbands)]
                # F = [np.log(np.mean(np.dot(self.csp[i].filters_, X[i]) ** 2, axis=1)) for i in range(self.nbands)]
                F = [np.log(np.var(np.dot(self.csp[i].filters_[:self.ncsp], X[i]), axis=1)) for i in range(self.nbands)]
                S = np.asarray([np.ravel(self.lda[i].transform(F[i].reshape(1, -1))) for i in range(self.nbands)]).T
                MS = np.log(self.norm_a.pdf(S)/self.norm_b.pdf(S))
                y_prob = self.clf.predict_proba(MS.reshape(1, -1))
                y_label = self.clf.predict(MS.reshape(1, -1))
            # print(y_prob, y_label)
            if out_param == 'prob': return y_prob
            elif out_param == 'label': return y_label
            
    def classify_set(self, Z, out_param='label'): # eeg_path, channels=None, 
        # self.load_eeg_test(eeg_path, channels)
        # smin, smax = int(self.tmin * self.Fs), int(self.tmax * self.Fs)
        # Z, t = extractEpochs(self.data_test, self.events_test, smin, smax, self.class_ids)
        if Z == []: return None
        else:
            bsize = 2 / (self.Fs / Z.shape[-1])
            if not self.is_sbcsp:
                X = self.filter.apply_filter(Z)
                if self.filt_type == 'DFT':  # extrai somente os bins referentes à banda de interesse
                    X[:, :, round(self.f_low*bsize):round(self.f_high*bsize)]
                F = self.csp.transform(X)
                # F = np.log(np.mean(np.dot(self.csp.filters_, X)**2, axis=2))
                # F = np.log(np.var(np.dot(self.csp.filters_, X), axis=2))
                y_prob = self.clf.predict_proba(F)
                y_label = self.clf.predict(F)
            else:
                if self.filt_type == 'DFT':
                    XF = self.filter.apply_filter(Z)
                    X = [XF[:,:, round(self.sub_bands[i][0]*bsize):round(self.sub_bands[i][1]*bsize)] for i in range(self.nbands)]
                elif self.filt_type in ['IIR','FIR']:
                    X = [self.filter[i].apply_filter(Z) for i in range(self.nbands)]
                
                # F = [np.log(np.mean(np.dot(self.csp[i].filters_, X[i])**2, axis=2)) for i in range(self.nbands)]
                # F = [np.log(np.var(np.dot(self.csp[i].filters_, X[i]), axis=2)) for i in range(self.nbands)]
                F = [self.csp[i].transform(X[i]) for i in range(self.nbands)]
                S = np.asarray([np.ravel(self.lda[i].transform(F[i])) for i in range(self.nbands)]).T
                MS = np.log(self.norm_a.pdf(S)/self.norm_b.pdf(S))
                y_prob = self.clf.predict_proba(MS)
                y_label = self.clf.predict(MS)
            # print(y_prob, y_label)
            if out_param == 'prob': return y_prob
            elif out_param == 'label': return y_label