# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import pickle
import itertools
import numpy as np
import pandas as pd
from time import time
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.linalg import eigh
from hyperopt import base, tpe, fmin, hp
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scripts.bci_utils import BCI, CSP, extractEpochs, nanCleaner

FINAL = []
CFG = 'sbcsp' # 'csp_lda' or 'sbcsp'
FILTRO = 'IIR' # 'IIR' or 'FFT'
fl = 8 
fh = 30
opt_iter = 1000

n_channels = 22
fs = 250
order = 5
t_start,t_end = 2.5,4.5
subjects = range(1,10)
classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

nf = fs/2.
bin0 = int(fl * (fs/nf))
binN = int(fh * (fs/nf))
W_Start = int(t_start * fs)
W_End = int(t_end * fs)

path = '/mnt/dados/eeg_path/IV2a/npy/'

""" Classes = LH, RH, FooT, TonGue
Nchannels = 22
Nsubjects = 9 (A01,A02,A03, ... ,A09)
Nsessions = 2 (_T, _E) 
            2 * 288 trials = 576 total trials -> 2*72 = 144 per class -> 4*72 = 288 per session
Fs= 250Hz   
Timestamps Protocol: startTrial=0; cue=2; startMI=3.25; endMI=6; endTrial=7.5-8.5
Samplestamps Protocol: startTrial=0; cue=500; startMI=813; endMI=1500; endTrial=1875
"""

def csp_lda(args):
    n_comp = args

    RES = []
    for cl in classes:
        for suj in subjects:   
            
            XT = np.load(path + str(suj) + 'T.npy')
            XV = np.load(path + str(suj) + 'E.npy')
            
            XT = ([XT[cl[0]-1], XT[cl[1]-1]])
            XV = ([XV[cl[0]-1], XV[cl[1]-1]])
            
            XT = np.concatenate([XT[0],XT[1]]) # Dados de treinamento das classes A, B
            XV = np.concatenate([XV[0],XV[1]]) # Dados de validação das classes A, B
            y = np.concatenate([np.zeros(int(len(XT)/2)), np.ones(int(len(XT)/2))]) # Vetor gabarito
            
            XTJ = XT[:,:,W_Start:W_End] 
            XVJ = XV[:,:,W_Start:W_End]
            
            start = time() # start timer
            
            if FILTRO == 'FFT':
                # Spectral filtering
                filtered = fft(XTJ)
                REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
                IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1)) 
                filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
                XTF = np.transpose(filtered, (1, 2, 0))       
                filtered = fft(XVJ)
                REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
                IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1))
                filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
                XVF = np.transpose(filtered, (1, 2, 0))
            
            elif FILTRO == 'IIR':
                # Temporal filtering
                b, a = butter(order, [fl/nf, fh/nf], btype='bandpass') # to filt IIR
                XTF = lfilter(b, a, XTJ)
                XVF = lfilter(b, a, XVJ)
            
            csp = CSP(n_components=n_comp)
            csp.fit(XTF, y)
            XT_CSP = csp.transform(XTF)
            XV_CSP = csp.transform(XVF) 
            
            clf = LinearDiscriminantAnalysis()
            #clf = SVC(kernel="poly", C=10**(-4))
            #clf = KNeighborsClassifier(n_neighbors=24, metric='minkowski', p=2) #minkowski e p=2 -> para usar distancia euclidiana padrão
            #clf = DecisionTreeClassifier(criterion='entropy', random_state=0) #max_depth = None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
            #clf = GaussianNB()
            #clf = MLPClassifier(verbose=False, max_iter=10000, tol=0.0001, activation='logistic', learning_rate_init=0.001, learning_rate='invscaling',  solver='adam') #hidden_layer_sizes=(100,),
               
            clf.fit(XT_CSP, y)
            scores = clf.predict(XV_CSP)
            
            cost = time() - start # stop timer (cost estimate)
            acc = np.mean(scores == y) 
            
            RES.append([suj,cl,acc,cost])
            # print (suj, cls, str(round(acc * 100, 2))+'%', str(int(cost*1000))+'ms')
    
    print(np.asarray(RES)[:,2].mean()*100, np.asarray(RES)[:,3].mean()*1000, np.asarray(RES)[:,3].sum())    
    return np.array(RES)

def sbcsp(args):
    print(args)
    fl, fh, n_comp, n_bands, Clog, = args
    #n_comp, n_bands, Clog = args
    
    bin0 = int(fl * (fs/nf))
    binN = int(fh * (fs/nf))
    
    RES = []
    for cl in classes:
        for suj in subjects:   
            
            XT = np.load(path + str(suj) + 'T.npy')
            XV = np.load(path + str(suj) + 'E.npy')
            
            XT = ([XT[cl[0]-1], XT[cl[1]-1]])
            XV = ([XV[cl[0]-1], XV[cl[1]-1]])
            
            XT = np.concatenate([XT[0],XT[1]]) # Dados de treinamento das classes A, B
            XV = np.concatenate([XV[0],XV[1]]) # Dados de validação das classes A, B
            y = np.concatenate([np.zeros(int(len(XT)/2)), np.ones(int(len(XT)/2))]) # Vetor gabarito
            
            XTJ = XT[:,:,W_Start:W_End] 
            XVJ = XV[:,:,W_Start:W_End]
            
            start = time() # start timer
            
            if FILTRO == 'FFT':
                # Spectral filtering
                filtered = fft(XTJ)
                REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
                IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1)) 
                filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
                XT_FFT = np.transpose(filtered, (1, 2, 0))       
                filtered = fft(XVJ)
                REAL = np.transpose(np.real(filtered)[:,:,bin0:binN], (2, 0, 1))
                IMAG = np.transpose(np.imag(filtered)[:,:,bin0:binN], (2, 0, 1))
                filtered = list(itertools.chain.from_iterable(zip(IMAG, REAL))) 
                XV_FFT = np.transpose(filtered, (1, 2, 0))
            
            # Divide sub-bands
            if FILTRO == 'FFT': n_bins = len(XT_FFT[0,0,:]) # ou (fh-fl) * 4 # Número total de bins de frequencia 
            elif FILTRO == 'IIR': n_bins = fh-fl 
            overlap = 2
            step = int(n_bins / n_bands)
            size = int(step * overlap) # tamanho fixo p/ todas sub bandas. overlap em 50%
            
            # Make sub-bands limits and Temporal/Spectral filtering
            SCORE_T = np.zeros((len(XT), n_bands))
            SCORE_V = np.zeros((len(XV), n_bands))
            for i in range(n_bands):
                if FILTRO == 'FFT':
                    # Only Spectral filtering
                    bin_ini = i*step
                    bin_fim = i*step+size
                    if bin_fim >= n_bins: bin_fim = n_bins-1
                    XTF = XT_FFT[:, :, bin_ini:bin_fim]
                    XVF = XV_FFT[:, :, bin_ini:bin_fim]
                elif FILTRO == 'IIR':
                # Only Temporal filtering
                    fl_sb = i*step+fl
                    fh_sb = i*step+size+fl
                    if fl_sb == 0: fl_sb = 0.001
                    if fh_sb > fh: fh_sb = fh
                    #print(fl_sb, fh_sb, nf, fl_sb/nf, fh_sb/nf)
                    
                    Wnl = fl_sb/nf
                    Wnh = fh_sb/nf
                    if Wnh >= 1: Wnh = 0.99
                    
                    b, a = butter(order, [Wnl, Wnh], btype='bandpass') # to filt IIR
                    XTF = lfilter(b, a, XTJ) 
                    XVF = lfilter(b, a, XVJ)
                
                csp = CSP(n_components=n_comp)
                csp.fit(XTF, y)
                XT_CSP = csp.transform(XTF) 
                XV_CSP = csp.transform(XVF)
        
                clf = LinearDiscriminantAnalysis()
                clf.fit(XT_CSP, y)
                SCORE_T[:, i] = np.ravel(clf.transform(XT_CSP)) # classificaçoes de cada época nas N sub bandas - auto validação
                SCORE_V[:, i] = np.ravel(clf.transform(XV_CSP)) # validação
            
            # Meta-classificador Bayesiano
            SCORE_T0 = SCORE_T[y == 0, :]
            m0 = np.mean(SCORE_T0, axis=0) #media classe A
            std0 = np.std(SCORE_T0, axis=0) #desvio padrão classe A
            
            SCORE_T1 = SCORE_T[y == 1, :]
            m1 = np.mean(SCORE_T1, axis=0)
            std1 = np.std(SCORE_T1, axis=0)
        
            p0 = norm(m0, std0) # p0 e p1 representam uma distribuição normal de médias m0 e m1, e desvio padrão std0 e std1
            p1 = norm(m1, std1)
            
            META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
            META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
            
            # SVM on top of the meta-classifier
            svc = SVC(kernel="linear", C=10**Clog)
            svc.fit(META_SCORE_T, y)
            scores = svc.predict(META_SCORE_V)
            
            cost = time() - start # stop timer (cost estimate)
            acc = np.mean(scores == y) 
        
            #print(suj, cl, n_comp, n_bands, str(round(acc * 100, 2))+'%', str(int(cost*1000))+'ms')
            RES.append([suj,cl,acc,cost])
    
    print(np.asarray(RES)[:,2].mean()*100, np.asarray(RES)[:,3].mean()*1000, np.asarray(RES)[:,3].sum())    
    return np.asarray(RES)

def f_objective(args):
    if CFG == 'csp_lda':
        n_comp = args
        args2 = (int(n_comp))
        result = csp_lda(args2)
    elif CFG == 'sbcsp':
        #n_comp, n_bands, Clog = args
        #args2 = (int(n_comp), int(n_bands), Clog)
        
        fl, fh, n_comp, n_bands, Clog = args
        args2 = (int(fl), int(fh), int(n_comp), int(n_bands), Clog)
        
        # n_comp, n_bands = args
        # args2 = (int(n_comp), int(n_bands), -4)    
        result = sbcsp(args2)
    global FINAL 
    FINAL = result
    acc_media = np.mean(result[:,2])
    #print(args,-acc_media)
    return -acc_media

if __name__ == "__main__": 
    ds = 'IV2a' # III3a, III4a, IV2a, IV2b, Lee19, LINCE
    auto_setup = True
    n_iter = 10
    crossval = False
    nfolds = 10
    test_perc = 0.1 if crossval else 0.5 
    overlap = True
    
    if ds == 'III3a':
        subjects = ['K3','K6','L1'] 
        classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    
    elif ds == 'III4a':
        subjects = ['aa','al','av','aw','ay']
        classes = [[1, 3]]
    
    elif ds == 'IV2a':        
        subjects = range(1,10) 
        classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] 
    
    elif ds == 'IV2b': 
        subjects = range(1,10)
        classes = [[1, 2]]
    
    elif ds == 'LINCE':
        subjects = ['CL_LR','CL_LF','TL_S1','TL_S2','WL_S1','WL_S2']
        classes = [[1, 2]]
    
    elif ds == 'Lee19':
        subjects = range(1, 55) 
        classes = [[1, 2]]
        cortex_only = True # True if only cortex channels is used
    
    subjects = [1] # uncomment to run one subject only
    for suj in subjects:
        # path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + '' + 'S' + str(suj) + 'sess2' + '.npy' #> ENTER THE PATH TO DATASET HERE (Lee19 default)
        path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + '' + 'A0' + str(suj)  + '.npy' #> ENTER THE PATH TO DATASET HERE  
        data, events, info = np.load(path_to_data, allow_pickle=True) # pickle.load(open(path_to_data, 'rb'))
        
        for class_ids in classes:
            
            max_knn_neig = int((info['trials_per_class'] * 2) * test_perc)
            # print(trials_per_class, max_knn_neig)
                                
            space = (
                hp.uniformint('fl', 8, 8), # hp.quniform('fl', 1, 20, 1),
                hp.uniformint('fh', 30, 30),  # hp.quniform('fh', 30, 49, 1),
                hp.quniform('tmin', 0.5, 0.5, 0.5),
                hp.quniform('tmax', 2.5, 2.5, 0.5),
                hp.quniform('ncomp', 2, info['eeg_channels'], 2),
                hp.choice('approach', [
                    {'option':'classic',},
                    # {'option':'sbcsp', 'nbands': hp.uniformint('nbands', 2, 48)} # hp.quniform('nbands', 2, max_nbands, 1)}
                    ]),
                hp.choice('filt', [
                    {'design':'DFT'},
                    # {'design':'IIR', 'iir_order': hp.uniformint('iir_order', 1, 8)}, #hp.quniform('iir_order', 1, 8, 1)},
                    ]),
                hp.choice('clf', [{'model':'SVM', 'C': hp.quniform('C', -4, -4, 1), 'kernel': hp.choice('kernel', [{'kf':'linear'}]),},])
                )
            
              
            bci = BCI(data, events, class_ids, overlap, info['fs'], crossval, nfolds, test_perc)
            
            
            trials = base.Trials()
        
        
        
            # if CFG == 'csp_lda':
            #     #print(f_objective((4))) # para teste unitário
            #     space = (hp.quniform('n_comp', 2, n_channels, 2)) # 2 até n_channels salto=2
            #     #space = ((hp.choice('n_comp', range(2, 50, 2))))
            
            # elif CFG == 'sbcsp': # ('n_bands', 1, (fh-fl)-1, 1) 
            #     #print(f_objective((5, 41, 2, 14, -3))) # para teste unitário
            #     space = (hp.quniform('fl', 0, 8, 1),
            #              hp.quniform('fh', 30, 51, 1),
            #              hp.quniform('n_comp', 2, n_channels, 2),
            #              hp.quniform('n_bands', 2, (fh-fl)-1, 1), 
            #              hp.quniform('Clog', -8, 4, 1))
            #     # space = (hp.quniform('n_comp', 2, n_channels, 2),
            #     #           hp.quniform('n_bands', 2, (fh-fl)-1, 1), 
            #     #           hp.quniform('Clog', -8, 4, 1))
            #               #hp.choice('Clog', [-4.])) #,hp.quniform('m', 20, 45, 1))
                
            # path_out = '/home/vboas/cloud/devto/res-hyperopt/opt_bci4-2a_' + CFG + '_' + FILTRO + '_' + str(fl) + '-' + str(fh)
            
            # try:
            #     print('Trying to pickle file')
            #     #trials = base.Trials()
            #     trials = pickle.load(open(path_out, 'rb'))
            # except:
            #     print('No trial file at specified path, creating new one')
            #     trials = base.Trials()
            # else:
            #     print('File found')
        
            try:
                print('Size of object: ' + str(len(trials)))
                # best = fmin(f_objective, space=space, algo=tpe.suggest, max_evals=len(trials) + opt_iter, trials=trials)
                best = fmin(bci.objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials)
                print(best)
                # pickle.dump(trials, open(path_out, 'wb'))
            except:
                print('Exception raised')
                #print(trials.)
                # pickle.dump(trials, open(path_out, 'wb'))
                raise
        
        
            # AAAA = pickle.load(open('/home/vboas/cloud/devto/res-hyperopt/opt_bci4-2a_sbcsp_FFT_4-40', 'rb'))
    
    
    
    
    
    