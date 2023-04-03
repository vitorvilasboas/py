# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:18:48 2020
@author: Vitor Vilas-Boas
"""

from hyperopt import pyll, hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.kde import gaussian_kde
# from tqdm import tqdm
# pbar = tqdm(total=max_evals, desc="Hyperopt")

# Let's plot the result of sampling from many different probability distributions:
hyperparam_generators = {
    # 'randint': hp.randint('randint', 5),
    'uniform': hp.uniformint('uniform', 2, 10),
    'max_depth_int': hp.qlognormal('max_depth_int', 3, 1, 1),
    'gamma_float' : hp.loguniform('gamma_float', -9, 3)
    # 'min_split': hp.qlognormal('min_split', 2, 1, 1),
    # 'fir_order': hp.quniform('fir_order', 2, 7, 1),
    #'loguniform': hp.loguniform('loguniform', 0.0001, 1),
    # 'normal': hp.normal('normal', 1, 2),
    #'lognormal': hp.lognormal('lognormal', 0.1, 0.3)
}


x = pyll.stochastic.sample(hyperparam_generators)



n_samples = 50
for title, space in hyperparam_generators.items():
    evaluated = [ pyll.stochastic.sample(space) for _ in range(n_samples) ]
    
    x_domain = np.linspace(min(evaluated), max(evaluated), n_samples)
    
plt.figure(figsize=(18,6))

hist = gaussian_kde(evaluated, 0.001)
plt.plot(x_domain, hist(x_domain), label="True Histogram")

blurred_hist = gaussian_kde(evaluated, 0.1)
plt.plot(x_domain, blurred_hist(x_domain), label="Smoothed Histogram")

plt.title("Histogram (pdf) for a {} distribution".format(title))
plt.legend()
plt.show()
    
#a = np.logspace(-9, 3, 13)