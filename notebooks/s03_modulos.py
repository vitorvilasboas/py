# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:58:34 2019
@author: Vitor Vilas-Boas
"""

import math
import math as matematica
from math import sqrt, floor
import random
import datetime
import time
import os


# Nativas: https://docs.python.org/3.7/library/index.html
# Extras PyPI: https://pypi.org/ (criadas pela comunidade)

start = time.time()
n = 1753.67

math.trunc(n) # truncate (elimina parte flutuante)

xplus = math.ceil(1.56) # arredonda para cima

xmin = math.floor(1.55) # arredonda para baixo

fat = math.factorial(4)

raiz = math.sqrt(9)

math.pow(4,3)

math.cos(3.141572)

aleatorio = random.random()

aleatorio = random.randint(1,10)

stop = time.time()

print(stop-start)

print(datetime.date.today())

print(datetime.date.isoformat(datetime.date.today()))

print(os.path.dirname(__file__))