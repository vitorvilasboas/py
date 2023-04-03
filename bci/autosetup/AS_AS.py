import copy
import pickle
import numpy as np
import pandas as pd
from hyperopt import base, fmin, tpe, hp
from time import time
from timeit import timeit

class AutoSetup:
    def __init__(self, setup=None, n_iter=100, load_last_setup=False, dump=False, path_dump=None):
        self.dump = dump
        self.load_last_setup = load_last_setup
        self.setup = setup
        self.n_iter = n_iter
        self.cost = []
        self.path_dump = path_dump 
        
        self.H = pd.DataFrame(columns=['fl','fh','tmin','tmax','ncsp','nbands','clf','acc','learner','cost'])

        max_knn_neig = 20 # int((self.setup.info_train['trials_per_class']  * 2) * self.setup.test_perc)
        self.space = (
            hp.uniformint('fl', 0, 15),
            hp.uniformint('fh', 20, 45),
            hp.quniform('tmin', 0, 2, 0.5),
            hp.quniform('tmax', 2, self.setup.info_train['trial_mi_time'], 0.5),
            hp.choice('ncsp', [2,4,6,8,len(self.setup.channels)]),
            hp.uniformint('nbands', 1, 25),
            hp.choice('clf', [
                {'model': 'LR'},
                {'model': 'LDA'},
                {'model': 'KNN', 'neig': hp.uniformint('neig', 2, max_knn_neig), }, 
                 # 'metric': hp.choice('metric', ['euclidean','chebyshev']),},
                {'model': 'SVM','C': hp.quniform('C', -6, 0, 1),}, 
                # {'kf': 'sigmoid'},
                # 'kernel': hp.choice('kernel', [{'kf': 'linear'}, {'kf': 'poly'}, {'kf': 'rbf'}]),
            ])
        )
    
    def run_optimizer(self):
        if self.load_last_setup:
            try:
                # print('Trying to pickle file')
                self.trials = pickle.load(open(self.path_dump, 'rb'))
            except:
                print('No trial file at specified path, creating new one')
                self.trials = base.Trials()
            else: print('File found')
        else:
            print('No load last trial file, creating new one')
            self.trials = base.Trials()

        try:
            print('Past trials: ' + str(len(self.trials)))
            init_vals = [{'fl': 4, 'fh': 40, 'tmin': 0.5, 'tmax': 2.5, 'ncsp': 8, 'nbands': 9, 'model': 'SVM', 'C': -4, 'kf': 'linear'}]
            self.best = fmin(self.validate, space=self.space, algo=tpe.suggest, max_evals=len(self.trials) + self.n_iter, trials=self.trials, points_to_evaluate=init_vals)
            if self.dump: pickle.dump(self.trials, open(self.path_dump, 'wb'))
            # print(self.best)
        except:
            print('Exception raised')
            if self.dump: pickle.dump(self.trials, open(self.path_dump, 'wb'))
            # print(self.trials.best_trial['misc']['vals'])
            raise
        # self.cost /= self.n_iter
        self.setup = None # clear setup data

    def validate(self, args):
        t0 = time()
        print(args)
        fl, fh, tmin, tmax, ncsp, nbands, clf = args
        while(tmax-tmin) < 1: tmax += 0.5  # garante janela mínima de 1seg

        if int(nbands) > 1:
            is_sbcsp = True
            nbands = int(fh-fl) if int(nbands) > int(fh-fl) else int(nbands)
        else: is_sbcsp = False; nbands = None
        self.setup.define_params(
            f_low=int(fl), f_high=int(fh), ncsp=int(ncsp), class_ids=self.setup.class_ids, tmin=tmin, tmax=tmax,
            fs=self.setup.Fs, filt_type=self.setup.filt_type, clf_dict=clf, filt_order=self.setup.filt_order, is_sbcsp=is_sbcsp,
            nbands=nbands, overlap=self.setup.overlap, crossval=self.setup.crossval, nfolds=self.setup.nfolds,
            test_perc=self.setup.test_perc)
        
        try: self.setup.process(single_eeg=True)
        except: self.setup.acc = 0.1
        
        learner = copy.deepcopy(self.setup)
        learner.clear_eeg_data()
        self.H.loc[len(self.H)] = [fl, fh, tmin, tmax, ncsp, nbands, clf, learner.acc, learner, time()-t0]
        
        return self.setup.acc * (-1)