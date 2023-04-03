import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import functools

class Regression(object):
    pass

class PolynomialFeature(object):
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree # degree : int - degree of polynomial

    def transform(self, x):
        if x.ndim == 1: x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()

class LinearRegression(Regression):
    def fit(self, X:np.ndarray, t:np.ndarray):
        self.w = np.linalg.pinv(X) @ t
        self.var = np.mean(np.square(X @ self.w - t))
        
    def predict(self, X:np.ndarray, return_std:bool=False):
        y = X @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y

class RidgeRegression(Regression):
    def __init__(self, lambd:float=1.):
        self.lambd = lambd # lambd = PRML alpha

    def fit(self, X:np.ndarray, t:np.ndarray):
        eye = np.eye(np.size(X, 1))
        self.w = np.linalg.solve(self.lambd * eye + X.T @ X, X.T @ t) 

    def predict(self, X:np.ndarray):
        return X @ self.w
    

if __name__ == "__init__":
    
    x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
    t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])
    
    x_correct = np.linspace(0, 1, 100)
    t_correct = np.sin(2 * np.pi * x_correct)
    
    x_test = np.linspace(0.01,0.99,1000, endpoint=False)
    noise = np.random.randn(len(x_test))
    # noise = np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
    t_test = np.sin(2*np.pi*x_test) + 0.1 * noise 
    
    M = [0, 1, 3, 9]
    
    for i, degree in enumerate(M):
        plt.subplot(2, 2, i + 1)
        feature = PolynomialFeature(degree)
        A_train = feature.transform(x_train)
        A_test = feature.transform(x_test)
        
        w1 = np.linalg.inv( A_train.T.dot(A_train)).dot(A_train.T.dot(t_train) )
        w2 = np.linalg.pinv(A_train) @ t_train
    
        model = LinearRegression()
        model.fit(A_train, t_train)
        y_test = model.predict(A_test)
    
        plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
        plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
        plt.plot(x_test, y_test, c="r", label="fitting")
        plt.ylim(-1.5, 1.5)
        plt.annotate("M={}".format(degree), xy=(-0.15, 1))
    plt.legend(bbox_to_anchor=(1.05, 0.64), loc=2, borderaxespad=0.)
    plt.show()
    
    ## Erro MSE
    training_errors = []
    test_errors = []
    
    # Erms = []
    # ErmsV = []
    
    for i in range(10):
        feature = PolynomialFeature(i)
        A_train = feature.transform(x_train)
        A_test = feature.transform(x_test)
    
        model = LinearRegression()
        model.fit(A_train, t_train)
        y_test = model.predict(A_test)
        y_train = model.predict(A_train)
        
        training_errors.append(np.sqrt(np.mean(np.square(y_train - t_train))))
        test_errors.append(np.sqrt(np.mean(np.square( y_test - t_test ))))
        
        # Erms.append( math.sqrt( (y_train-t_train).T.dot(y_train-t_train)/len(x_train) ) )
        # ErmsV.append( math.sqrt( (y_test - t_test).T.dot(y_test - t_test)/len(x_test) ) )
        
        
    plt.plot(training_errors, 'o-', mfc="none", mec="b", ms=10, c="b", label="Training")
    plt.plot(test_errors, 'o-', mfc="none", mec="r", ms=10, c="r", label="Test")
    plt.legend()
    plt.xlabel("degree")
    plt.ylabel("RMSE")
    plt.show()
    
    
    ## Regularization (quanto maior o expoente, maior a regularizacao)
    
    l = np.exp(-18) # np.exp(-18) == 1e-8
    l = np.exp(-6.9) # 1e-3 == 0.001 ~ np.exp(-6.9)# 
    
    feature = PolynomialFeature(9)
    A_train = feature.transform(x_train)
    A_test = feature.transform(x_test)
    
    model = RidgeRegression(lambd=l) 
    model.fit(A_train, t_train)
    y_test = model.predict(A_test)
    
    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y_test, c="r", label="fitting")
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.annotate("M=9", xy=(-0.15, 1))
    plt.show()    
    
    
    
# =============================================================================
#     M = 3
#     lambd = -18
#     
#     A_train = np.asarray([x_train**i for i in range(M+1)]).T
#     w = np.linalg.inv( A_train.T.dot(A_train) + np.exp(lambd) * np.identity(M+1) ).dot(
#             A_train.T.dot(t_train) )
#     y_train = np.dot(A_train,w)
#     
#     A_test = np.asarray([x_test**i for i in range(M+1)]).T
#     y_test = np.dot(A_test,w)
#     
#     plt.figure(figsize=(13,5))
#     plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="target train")
#     #plt.scatter(x_train, y_train, facecolor="none", edgecolor="r", s=50, label="predicted train $y(x,w)$")
#     plt.plot(x_test, y_test, c="r", label="$y(x,w)$") #curve of predicted test data 
#     plt.plot(x_correct, t_correct, c="g", alpha=0.6, label="$\sin(2\pi x)$")
#     
#     plt.ylabel('$t$', fontsize=13, rotation = 360)
#     plt.xlabel('$\mathit{x}$', fontsize=13,labelpad=-15)
#     plt.xticks([0,1])
#     plt.yticks([-1,0,1])
#     plt.ylim(-1.5,1.5)
#     plt.annotate(('ln $\lambda$ = {}'.format(lambd)), xy=(0.8, 1))
#     plt.legend(loc='lower left')
#     #plt.savefig('figures/Fig2.1.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
# 
#     E_train = (y_train-t_train).T.dot(y_train-t_train) + np.exp(lambd)/2 * w.T.dot(w)
#     Erms_train = math.sqrt(E_train/len(x_train))
#     
#     E_test = (y_test-t_test).T.dot(y_test-t_test) + np.exp(lambd)/2 * w.T.dot(w)
#     Erms_test = math.sqrt(E_test/len(x_test))
# =============================================================================
