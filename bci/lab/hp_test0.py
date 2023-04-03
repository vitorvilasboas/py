#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:50:25 2020
@author: vboas
"""
from functools import partial
from hyperopt.pyll import scope

from hyperopt import hp, fmin, rand, tpe, space_eval
from hyperopt.pyll.stochastic import sample

space = [hp.normal('a', 0.5, 0.25)]
space2= [hp.loguniform('a', -7, 0)]
space3= [hp.quniform('a', 2, 20, 2)]
space4= [hp.uniform('a', 2, 20)]
space5= [hp.uniformint('a', 2, 20, 2)]
print(sample(space4))

def q(args):
    print(args)
    x, y = args
    return x ** 2 + y ** 2

space = [hp.uniform('x', 0, 1), hp.normal('y', 0, 1)]

# best = fmin(q, space, algo=rand.suggest, max_evals=2)
# print(best)
# print(space_eval(space, best))

# best = fmin(q, space, algo=tpe.suggest, max_evals=2)
# print(best)
# print(space_eval(space, best))

# algo = partial(tpe.suggest, n_startup_jobs=10)
# best = fmin(q, space, algo=algo, max_evals=2)
# print(best)
# print(space_eval(space, best))



# list_space = [hp.uniform('a', 0, 1), hp.loguniform('b', 0, 1)]
# tuple_space = (hp.uniform('a', 0, 1), hp.loguniform('b', 0, 1))
# dict_space = {'a': hp.uniform('a', 0, 1), 'b': hp.loguniform('b', 0, 1)}
# nested_space = [
#     [{'case': 1, 'a': hp.uniform('a', 0, 1)},
#      {'case': 2, 'b': hp.loguniform('b', 0, 1)}],
#     'extra literal string', hp.randint('r', 10)]

# print(sample(list_space))
# print(sample(nested_space))

# # @scope.define
# def foo(x):
#     return str(x) * 3

# # print(foo(0))

# expr_space = {'f': hp.loguniform('b', 0, 1), 
#               'b': scope.minimum(hp.loguniform('b', 0, 1), 10),
#               'c': scope.foo(hp.randint('cbase', 5)),
#               'a': 1 + hp.uniform('a', hp.uniform('g', 0, 5), 1),
#               }

# print(sample(expr_space))

# best = fmin(q, expr_space, algo=rand.suggest, max_evals=2)
# print(best)

