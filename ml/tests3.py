import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import functools
from scipy.stats import multivariate_normal

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


class GaussianFeature(object):
    """ gaussian function = exp(-0.5 * (x - m) / v) """
    def __init__(self, mean, var):
        print(mean)
        """ mean : (n_features, ndim) or (n_features,) ndarray    - locais para localizar a função gaussiana
            var : float                                           - variance of the gaussian function """
        if mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2
        assert isinstance(var, float) or isinstance(var, int)
        self.mean = mean
        self.var = var

    def _gauss(self, x, mean):
        return np.exp(-0.5 * np.sum(np.square(x - mean), axis=-1) / self.var)

    def transform(self, x):
        """ input array x : (sample_size, ndim) or (sample_size,)   
            output gaussian feature : (sample_size, n_features) """
        if x.ndim == 1:
            x = x[:, None]
        else:
            assert x.ndim == 2
        assert np.size(x, 1) == np.size(self.mean, 1)
        basis = [np.ones(len(x))]
        for m in self.mean:
            basis.append(self._gauss(x, m))
        return np.asarray(basis).transpose()


class SigmoidalFeature(object):
    """ Sigmoidal features = 1 / (1 + exp((m - x) @ c) """
    def __init__(self, mean, coef=1):
        """ mean : (n_features, ndim) or (n_features,) ndarray    - center of sigmoid function
            coef : (ndim,) ndarray or int or float                - coefficient to be multplied with the distance """
        if mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2
        if isinstance(coef, int) or isinstance(coef, float):
            if np.size(mean, 1) == 1:
                coef = np.array([coef])
            else:
                raise ValueError("mismatch of dimension")
        else:
            assert coef.ndim == 1
            assert np.size(mean, 1) == len(coef)
        self.mean = mean
        self.coef = coef

    def _sigmoid(self, x, mean):
        return np.tanh((x - mean) @ self.coef * 0.5) * 0.5 + 0.5

    def transform(self, x):
        """ input array x : (sample_size, ndim) or (sample_size,) ndarray
            output sigmoidal features : (sample_size, n_features) ndarray """
        if x.ndim == 1:
            x = x[:, None]
        else:
            assert x.ndim == 2
        assert np.size(x, 1) == np.size(self.mean, 1)
        basis = [np.ones(len(x))]
        for m in self.mean:
            basis.append(self._sigmoid(x, m))
        return np.asarray(basis).transpose()


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
    

class BayesianRegression(Regression):
    """ w ~ N(w|0, alpha^(-1)I)
        y = X @ w
        t ~ N(t|X @ w, beta^(-1)) """
    def __init__(self, alpha:float=1., beta:float=1.):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _is_prior_defined(self) -> bool:
        return self.w_mean is not None and self.w_precision is not None

    def _get_prior(self, ndim:int) -> tuple:
        if self._is_prior_defined():
            return self.w_mean, self.w_precision
        else:
            return np.zeros(ndim), self.alpha * np.eye(ndim)

    def fit(self, X:np.ndarray, t:np.ndarray):
        """ bayesian update of parameters given training dataset
            X : (N, n_features) np.ndarray      - training data independent variable
            t : (N,) np.ndarray                 - training data dependent variable """
        mean_prev, precision_prev = self._get_prior(np.size(X, 1))

        w_precision = precision_prev + self.beta * X.T @ X
        w_mean = np.linalg.solve(
            w_precision,
            precision_prev @ mean_prev + self.beta * X.T @ t
        )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)

    def predict(self, X:np.ndarray, return_std:bool=False, sample_size:int=None):
        """ return mean (and standard deviation) of predictive distribution
            X : (N, n_features) np.ndarray      - independent variable
            return_std : bool, optional         - flag to return standard deviation (the default is False)
            sample_size : int, optional         - number of samples to draw from the predictive distribution
                                                  (the default is None, no sampling from the distribution)
        Returns:
            y : (N,) np.ndarray                         - mean of the predictive distribution
            y_std : (N,) np.ndarray                     - standard deviation of the predictive distribution
            y_sample : (N, sample_size) np.ndarray      - samples from the predictive distribution """
        if sample_size is not None:
            w_sample = np.random.multivariate_normal(
                self.w_mean, self.w_cov, size=sample_size
            )
            y_sample = X @ w_sample.T
            return y_sample
        y = X @ self.w_mean
        if return_std:
            y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y


class EmpiricalBayesRegression(BayesianRegression):
    """ Modelo de regressão empírica de Bayes, também conhecido como tipo 2, máxima verossimilhança, 
        máxima verossimilhança generalizada, aproximação de evidências
    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    evidence function p(t|X,alpha,beta) = S p(t|w;X,beta)p(w|0;alpha) dw """
    
    def __init__(self, alpha:float=1., beta:float=1.):
        super().__init__(alpha, beta)

    def fit(self, X:np.ndarray, t:np.ndarray, max_iter:int=100):
        """maximization of evidence function with respect to the hyperparameters alpha and beta given training dataset
           X : (N, D) np.ndarray        - training independent variable
           t : (N,) np.ndarray          - training dependent variable
           max_iter : int               - maximum number of iteration """
        M = X.T @ X
        eigenvalues = np.linalg.eigvalsh(M)
        eye = np.eye(np.size(X, 1))
        N = len(t)
        for _ in range(max_iter):
            params = [self.alpha, self.beta]

            w_precision = self.alpha * eye + self.beta * X.T @ X
            w_mean = self.beta * np.linalg.solve(w_precision, X.T @ t)

            gamma = np.sum(eigenvalues / (self.alpha + eigenvalues))
            self.alpha = float(gamma / np.sum(w_mean ** 2).clip(min=1e-10))
            self.beta = float(
                (N - gamma) / np.sum(np.square(t - X @ w_mean))
            )
            if np.allclose(params, [self.alpha, self.beta]):
                break
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(w_precision)

    def _log_prior(self, w):
        return -0.5 * self.alpha * np.sum(w ** 2)

    def _log_likelihood(self, X, t, w):
        return -0.5 * self.beta * np.square(t - X @ w).sum()

    def _log_posterior(self, X, t, w):
        return self._log_likelihood(X, t, w) + self._log_prior(w)

    def log_evidence(self, X:np.ndarray, t:np.ndarray):
        """ logarithm or the evidence function
            X : (N, D) np.ndarray           - indenpendent variable
            t : (N,) np.ndarray             - dependent variable
            Returns: float  log evidence """
        N = len(t)
        D = np.size(X, 1)
        return 0.5 * (
            D * np.log(self.alpha) + N * np.log(self.beta)
            - np.linalg.slogdet(self.w_precision)[1] - D * np.log(2 * np.pi)
        ) + self._log_posterior(X, t, self.w_mean)

        

     
if __name__ == "__main__":
    
    np.random.seed(1234)
    
    
    x_correct = np.linspace(0, 1, 100)
    t_correct = np.sin(2 * np.pi * x_correct)
    
    x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
    t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ])
    
    x_test = np.linspace(0.01,0.99,100, endpoint=False)
    noise = np.random.randn(len(x_test))
    # noise = np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
    t_test = np.sin(2*np.pi*x_test) + 0.1 * noise 
    
    x = np.linspace(-1, 1, 100)
    
    M = 3 # complexity or n_features
     
    ########## 3.1 Linear Basis Function Models ###########
    
    A_polynomial = PolynomialFeature(M).transform(x[:, None])
    
    mu = np.linspace(-1, 1, M)
    var = 0.1 # np.var(x) #s
    
    A_gaussian = GaussianFeature(mu, var).transform(x)
    
    A_sigmoidal = SigmoidalFeature(mu, M-1).transform(x)

    plt.figure(figsize=(20, 5))
    for i, A in enumerate([A_polynomial, A_gaussian, A_sigmoidal]):
        plt.subplot(1, 3, i + 1)
        for j in range(M+1): plt.plot(x, A[:, j])
        
        
    ########## 3.1.1 Maximum likelihood and least squares ###########
    
    # Pick one of the three features below
    # feature = PolynomialFeature(8)
    # feature = GaussianFeature(np.linspace(0, 1, 3), 0.1)
    feature = GaussianFeature(np.linspace(x_train[0], x_train[-1], 9), np.var(x_train))
    # feature = SigmoidalFeature(np.linspace(0, 1, 8), 10)
    
    A_train = feature.transform(x_train)
    A_test = feature.transform(x_test)
    
    model = LinearRegression()
    model.fit(A_train, t_train)
    
    y, y_std = model.predict(A_test, return_std=True)
    
    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_correct, t_correct, label="$\sin(2\pi x)$")
    plt.plot(x_test, y, label="prediction")
    plt.fill_between( 
            x_correct, y - y_std, y + y_std,
            color="orange", alpha=0.5, label="std.")
    plt.legend()
    plt.ylim(-1.5,1.5)
    plt.show()
    
    
    
    
    M = 3
    
#    interv = int(len(x_train)/M) # calculando intervalo de indices entre os pontos médios x_train 
#    pos = [ interv*i for i in range(1,M+1) ] # definindo os indices dos pontos médios de x_train a partir dos intervalos
#    mu = x_train[pos] # capturando os valores dos pontos médios de x_train para a gaussiana
    mu = np.linspace(x_train[0], x_train[-1], M) # oooouuuuuu usar o linspace para tudo isso :]

    # faixa = [ x_train[pos[i]:pos[i+1]] for i in range(0,len(pos)) ]
    var = np.var(x_train) #s
    
#    feature = GaussianFeature(mu, var)
#    A_train = feature.transform(x_train)
#    A_test = feature.transform(x_test)
    

    x_train1 = x_train[:, None]
    mu = mu[:, None]      
    #assert np.size(x_train1, 1) == np.size(mu, 1)
    base = [np.ones(len(x_train1))]
    for m in mu: base.append(np.exp(-0.5 * np.sum(np.square(x_train1 - m), axis=-1) / var))
    A_train = np.asarray(base).T
    
    x_test1 = x_test[:, None]     
    #assert np.size(x_test1, 1) == np.size(mu, 1)
    base = [np.ones(len(x_test1))]
    for m in mu: base.append(np.exp(-0.5 * np.sum(np.square(x_test1 - m), axis=-1) / var))
    A_test = np.asarray(base).T
    
    w = np.linalg.pinv(A_train) @ t_train
    
    y_train = A_train @ w
   
    Err_train = np.mean(np.square(y_train - t_train)) # bias decomposition
    
    y_test = A_test @ w
    
    Err_test = np.mean(np.square(y_train - t_train)) 
    
    y_test_rmse = np.sqrt(Err_train) + np.zeros_like(y_test) # variance decomposition
    
    
    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_correct, t_correct, label="$\sin(2\pi x)$")
    plt.plot(x_test, y_test, label="prediction")
    plt.fill_between( 
            x_correct, y_test - y_test_rmse, y_test + y_test_rmse,
            color="orange", alpha=0.5, label="std.")
    plt.legend()
    plt.ylim(-1.5,1.5)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
