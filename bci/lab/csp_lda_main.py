# -*- coding: utf-8 -*-
import numpy as np

import csp_lda_fft_depois as fft
import csp_lda_proj_depois as proj
import csp_lda_butter_depois as buttpos
import csp_lda_butter_primeiro as buttpre

classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
sujeito = 1
atraso = 3
n_componetes = 6
fl = 8 
fh = 30
ordem = 5

res = [fft.csp_lda(suj, classes[i], atraso, n_componetes, fl, fh, ordem) for suj in range(1, 10) for i in range(0, 6)]
print('FFT:\t', str(round(np.mean(np.asarray(res)[:,0])*100, 2)) , str(round(sum(np.asarray(res)[:,1]), 2)) , str(round(np.mean(np.asarray(res)[:,1]), 2)))

res = [proj.csp_lda(suj, classes[i], atraso, n_componetes, fl, fh, ordem) for suj in range(1, 10) for i in range(0, 6)]
print('Proj:\t', str(round(np.mean(np.asarray(res)[:,0])*100, 2)) , str(round(sum(np.asarray(res)[:,1]), 2)) , str(round(np.mean(np.asarray(res)[:,1]), 2)))

res = [buttpos.csp_lda(suj, classes[i], atraso, n_componetes, fl, fh, ordem) for suj in range(1, 10) for i in range(0, 6)]
print('IIRpos:\t', str(round(np.mean(np.asarray(res)[:,0])*100, 2)) , str(round(sum(np.asarray(res)[:,1]), 2)) , str(round(np.mean(np.asarray(res)[:,1]), 2)))

res = [buttpre.csp_lda(suj, classes[i], atraso, n_componetes, fl, fh, ordem) for suj in range(1, 10) for i in range(0, 6)]
print('IIRpre:\t', str(round(np.mean(np.asarray(res)[:,0])*100, 2)) , str(round(sum(np.asarray(res)[:,1]), 2)) , str(round(np.mean(np.asarray(res)[:,1]), 2)))
