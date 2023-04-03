import copy
import pickle
import numpy as np
import pandas as pd
from hyperopt import base, fmin, tpe, hp

class AutoSetup:
    def __init__(self, setup=None, n_iter=100, load_last_setup=False):
        self.load_last_setup = load_last_setup
        self.setup = setup
        self.n_iter = n_iter
        
        self.H = pd.DataFrame(columns=['fl','fh','tmin','tmax','ncsp','nbands','clf','acc','learner'])

        max_knn_neig = int((self.setup.info_train['trials_per_class']  * 2) * self.setup.test_perc)

        # fl_ = hp.uniformint("fl", 0, 10)  # 0, 15
        # fh_ = hp.uniformint("fh", 10, 25)
        self.space = (
            # {"fl": fl_},
            # {"fh": (fh_ + fl_)},  # fl_ + 20, # hp.uniform("fh", (10 + fl_), 40),
            hp.uniformint('fl', 0, 15),
            hp.uniformint('fh', 20, 45),
            hp.quniform('tmin', 0, 2, 0.5),
            hp.quniform('tmax', 2, self.setup.info_train['trial_mi_time'], 0.5),
            hp.quniform('ncsp', 2, len(self.setup.channels), 2),  # 21 #14 #116
            # hp.choice('ncsp', [2, 4, 6, 8, 20, len(self.setup.channels)]),
            hp.choice('approach', [
                {'option': 'classic', },
                {'option': 'sbcsp', 'nbands': hp.uniformint('nbands', 2, 25)}, # hp.quniform('nbands', 1, ((fh_+fl_)-fl_)//2, 1)
            ]),
            hp.choice('filt', [
                {'design': 'DFT'},
                # {'design':'IIR', 'iir_order': hp.uniformint('iir_order', 1, 8)},
                # {'design':'FIR', 'fir_order': hp.uniformint('fir_order', 2, 7)},
            ]),
            # hp.pchoice('clf', [
            #     (0.25, {'model':'LDA'}),
            #     (0.25, {'model':'SVM','C': hp.quniform('C', -6, 0, 1), # hp.loguniform('C', -6, 0), #
            #       'kernel':hp.choice('kernel',[{'kf':'linear'},{'kf':'poly'},{'kf':'sigmoid'},{'kf':'rbf'}])}),
            #     (0.25, {'model':'KNN','neig':hp.uniformint('neig', 5, 20),
            #       'metric':hp.choice('metric',['euclidean','manhattan','minkowski','chebyshev'])}),
            #     (0.25, {'model':'LR'})
            #     ]),
            hp.choice('clf', [
                {'model': 'LR'},
                {'model': 'LDA', 'lda_solver': hp.choice('lda_solver', ['svd', 'lsqr', 'eigen']),  #
                 # 'shrinkage': hp.choice('shrinkage', [None, 'auto', {'shrinkage_float':  hp.uniform('shrinkage_float', 0, 1)}]) #np.logspace(-4, 0, 1)
                 },
                {'model': 'KNN',
                 'neig': hp.uniformint('neig', 2, max_knn_neig),  # 'neig': hp.quniform('neig', 2, max_knn_neig, 1),
                 'metric': hp.choice('metric', ['euclidean', 'manhattan', 'minkowski', 'chebyshev']),
                 # {'mf':'cityblock'}, {'mf':'cosine'}, {'mf':'l1'}, {'mf':'l2'},
                 # 'p': hp.quniform('p', 2, 50, 1)
                 },
                {'model': 'SVM',
                 'C': hp.quniform('C', -6, 0, 1),
                 # np.logspace(-8, 4, 13), # hp.quniform('C', -8, 4, 1) hp.lognormal('C', 0, 1),
                 'kernel': hp.choice('kernel', [{'kf': 'linear'}, {'kf': 'poly'}, {'kf': 'sigmoid'}, {'kf': 'rbf'}]),
                 # , 'width': hp.lognormal('width', 0, 1) # 'degree': hp.uniformint('degree', 2, 3)
                 # 'gamma': hp.choice('gamma', ['scale', 'auto', {'gamma_float': hp.quniform('gamma_float', -9, 4, 1)}]), # hp.loguniform('gamma_float', -9, 3)  np.logspace(-9, 3, 13)),
                 },
                {'model': 'MLP',
                 'eta': hp.quniform('eta', -5, -2, 1),
                 # hp.quniform('eta', 0.0001, 0.1, 0.0001)    hp.choice('eta', [0.1,0.01,0.001,0.0001]),
                 'n_neurons': hp.quniform('n_neurons', 10, 200, 10),  # hp.uniformint('n_neurons', 50, 500),
                 'n_hidden': hp.uniformint('n_hidden', 1, 3),
                 'activ': hp.choice('activ', [{'af': 'identity'}, {'af': 'logistic'}, {'af': 'tanh'}, {'af': 'relu'}]),
                 'mlp_solver': hp.choice('mlp_solver', ['adam', 'lbfgs', 'sgd']),
                 # 'alpha': hp.quniform('alpha', -8, 1, 1),  # hp.lognormal('alpha', 0, 1),
                 # 'eta_type': hp.choice('eta_type', ['constant', 'invscaling', 'adaptive']),
                 },
                # {'model': 'Bayes'},
                # {'model': 'DTree',
                #  'crit': hp.choice('crit', ['gini', 'entropy']),
                #  # 'max_depth': hp.choice('max_depth', [None, {'max_depth_int': hp.qlognormal('max_depth_int', 3, 1, 1)}]), # np.random.lognormal(3, 1, 1) ]),
                #  # 'min_split': hp.uniform('min_split', 0.0001, 1), #  np.random.lognormal(2, 1, 1) # hp.qlognormal('min_split', 2, 1, 1)
                #  }
            ])
        )

    def run_optimizer(self):
        if self.load_last_setup:
            try:
                # print('Trying to pickle file')
                self.trials = pickle.load(open('.' + '/trials_setup.pkl', 'rb'))
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
            # pickle.dump(self.trials, open(self.session.info.nickname + '/trials_setup.pkl', 'wb'))
            # print(self.best)
        except:
            print('Exception raised')
            # pickle.dump(self.trials, open(PATH_TO_SESSION + self.session.info.nickname + '/trials_setup.pkl', 'wb'))
            # print(self.trials.best_trial['misc']['vals'])
            raise
        self.setup = None # clear setup data

    def validate(self, args):
        # print(args)
        fl, fh, tmin, tmax, ncsp, approach, filt, clf = args
        # fl, fh = int(fl['fl']), int(fh['fh'])
        # print([fl, fh, tmin, tmax, ncsp, approach, filt, clf])

        while(tmax-tmin) < 1: tmax += 0.5  # garante janela mínima de 1seg

        if approach['option'] == 'sbcsp':
            is_sbcsp = True
            nbands = int(fh-fl) if int(approach['nbands']) > int(fh-fl) else int(approach['nbands'])
        else: is_sbcsp = False; nbands = None

        forder = None if filt['design'] == 'DFT' else filt['iir_order'] if filt['design'] == 'IIR' else filt['fir_order']

        self.setup.define_params(
            f_low=int(fl), f_high=int(fh), ncsp=int(ncsp), class_ids=self.setup.class_ids, tmin=tmin, tmax=tmax,
            fs=self.setup.Fs, filt_type=filt['design'], clf_dict=clf, filt_order=forder, is_sbcsp=is_sbcsp,
            nbands=nbands, overlap=self.setup.overlap, crossval=self.setup.crossval, nfolds=self.setup.nfolds,
            test_perc=self.setup.test_perc)
        self.setup.process(single_eeg=True)
        
        learner = copy.deepcopy(self.setup)
        learner.clear_eeg_data()
        self.H.loc[len(self.H)] = [fl, fh, tmin, tmax, ncsp, nbands, clf, self.setup.acc, learner]
        
        return self.setup.acc * (-1)

