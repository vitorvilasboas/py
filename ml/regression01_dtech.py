# -*- coding: utf-8 -*-

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# gerando uma massa de dados:
x, y = make_regression(n_samples=200,n_features=3,noise=0.3)

plt.scatter(x[:,0],y)
plt.show()

modelo = LinearRegression(1)

modelo.fit(x[:,0],y)

b = modelo.intercept_

a = modelo.coef_