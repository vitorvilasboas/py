# -*- coding: utf-8 -*-
import pandas as pd
import sklearn.datasets as ds
from scipy.io import loadmat

iris = ds.load_iris()

X = iris.data

y = iris.target

iris_csv = pd.read_csv('/home/vboas/cloud/devto/MLcourse/iris.csv')
X = iris_csv.iloc[:, 0:4].values
classe = iris_csv.iloc[:, 4].values

iris_mat = loadmat('../dataIRIS.mat')['data']  
features = iris_mat[:, 0:4]
saidas = iris_mat[:, 4]
# -*- coding: utf-8 -*-