# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

def posterior(Phi, t, alpha, beta, return_inverse=False):
    """Computes mean and covariance matrix of the posterior distribution."""
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(t)

    if return_inverse: return m_N, S_N, S_N_inv
    else: return m_N, S_N


def posterior_predictive(Phi_test, m_N, S_N, beta):
    """Computes mean and variances of the posterior predictive distribution."""
    y = Phi_test.dot(m_N)
    # Only compute variances (diagonal elements of covariance matrix)
    y_var = 1 / beta + np.sum(Phi_test.dot(S_N) * Phi_test, axis=1)
    
    return y, y_var

def gaussian_basis_function(x, mu, sigma=0.1):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)

def polynomial_basis_function(x, degree):
    return x ** degree

def expand(x, bf, bf_args=None):
    if bf_args is None:
        return np.concatenate([np.ones(x.shape), bf(x)], axis=1)
    else:
        return np.concatenate([np.ones(x.shape)] + [bf(x, bf_arg) for bf_arg in bf_args], axis=1)


if __name__ == "__init__":
    x_correct = np.linspace(0, 1, 100)
    t_correct = np.sin(2 * np.pi * x_correct)
    
    x_train = np.array([0.1387, 0.2691, 0.3077, 0.3625, 0.4756, 0.5039, 0.5607, 0.6468, 0.7490, 0.7881 ])
    t_train = np.array([0.8260, 1.0469, 0.7904, 0.6638, 0.1731, -0.0592, -0.2433, -0.6630, -1.0581, -0.8839 ]) # variável dependente treinamento
    
    x_test = np.linspace(0.1,0.9,100, endpoint=False) #0.1-0.8
    noise = np.random.randn(len(x_test)) # np.random.normal(scale=0.3, size=x_train.shape) #scale=desvio padrão
    t_test = np.sin(2*np.pi*x_test) + 0.1 * noise
    # t_test = -0.3 + 0.5 * X + np.random.normal(scale=np.sqrt(1/beta), size=X.shape)
    
    M = 3
    
    A_train = np.asarray([x_train**i for i in range(M+1)]).T
    A_test = np.asarray([x_test**i for i in range(M+1)]).T
    
    
    # Modelo de Regressão Bayesiano
    # w ~ N(w|0, lambd^(-1) * I)
    # y = A @ w
    # t ~ N(t|A @ w, beta^(-1))
    
    
    
    
    M = 9
    A_train = np.asarray([x_train**i for i in range(M+1)]).T # variável independente treinamento
    
    A_test = np.asarray([x_test**i for i in range(M+1)]).T # variável independente teste
    
    lambd = 2e-3
    beta = 2
    
    ndim = np.size(A_train, 1)
    mean_prev, precision_prev = np.zeros(ndim), lambd * np.eye(ndim)
    w_precision = precision_prev + beta * A_train.T @ A_train
    w_mean = np.linalg.solve( w_precision, precision_prev @ mean_prev + beta * A_train.T @ t_train )
    w_cov = np.linalg.inv(w_precision)
    
    # Calculando a média e o desvio padrão da distribuição preditiva (dp)
    # w_sample = np.random.multivariate_normal(w_mean, w_cov, size=sample_size ) # if sample_size is not None: - número de amostras a serem retiradas da distribuição preditiva
    # y_sample = A_test @ w_sample.T   # amostras da dp
    y = A_test @ w_mean # média da dp
    y_var = 1 / beta + np.sum(A_test @ w_cov * A_test, axis=1) 
    y_err = np.sqrt(y_var) # y_std = y_err  # desvio padrao da dp
    
    plt.figure(figsize=(8,5))
    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_test, t_test, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y, c="r", label="mean")
    plt.fill_between(x_test, y - y_err, y + y_err, color="pink", label="std.", alpha=0.5)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1.5, 1.5)
    plt.annotate("M=9", xy=(0.8, 1))
    plt.legend(bbox_to_anchor=(1.05, 1.), loc='best', borderaxespad=0.)
    
    
    #x_test = np.arange(-10, 10, 0.001)
    y2 = norm.pdf(x_test,y,y_err)
    plt.plot(y2, x_test)
    #plt.fill_between(y2,x_test,0, alpha=0.3, color='b')
    
    # -------------------------------
    
    x = [8,2,10,10]
    x = np.linspace(0,1,1000)
    ruido = np.random.normal(scale=0.25, size=len(x)) # np.random.randn(len(x)) 
    x = np.sin(2*np.pi*x) + 0.1*ruido
    
    # Variância e desvio padrão são medidas de variabilidade e medem a dispersão (dos 
    # dados em relação à média
    # variancia = soma dos quadrados das diferenças entre cada observação e a média 
    # das observações dividido pelo número de observações
    var1 = np.sum(np.asarray(x-np.mean(x))**2)/len(x) 
    var2 = np.var(x)
    
    # desvio padrão = 
    std1 = math.sqrt(var1)
    std2 = np.std(x)
    
    normal1 = ( 1/math.sqrt(2*np.pi*var1) ) * np.exp( (-1/(2*var1))*((x-np.mean(x))**2)  )
    
    normal2 = norm.pdf(x, np.mean(x), np.std(x))
     
    
    # -------------------------------
    A = np.asarray([x_train**i for i in range(M+1)]).T
    w = np.linalg.inv( (A.T).dot(A) + (np.exp(-18) * np.identity(M+1)) ).dot( (A.T).dot(t_train) )
    y = np.dot(A,w)
    # np.var(x)
    #plt.plot(xv,norm.pdf(yv, yv.mean(), yv.std()))
    
    x = [8,2,10,10]
    x = np.linspace(0,1,1000)
    ruido = np.random.normal(scale=0.25, size=len(x)) # np.random.randn(len(x)) 
    x = np.sin(2*np.pi*x) + 0.1*ruido
    
    # Variância e desvio padrão são medidas de variabilidade e medem a dispersão (dos 
    # dados em relação à média
    # variancia = soma dos quadrados das diferenças entre cada observação e a média 
    # das observações dividido pelo número de observações
    var1 = np.sum(np.asarray(x-np.mean(x))**2)/len(x) 
    var2 = np.var(x)
    
    # desvio padrão = 
    std1 = math.sqrt(var1)
    std2 = np.std(x)
    
    normal1 = ( 1/math.sqrt(2*np.pi*var1) ) * np.exp( (-1/(2*var1))*((x-np.mean(x))**2)  )
    
    normal2 = norm.pdf(x, np.mean(x), np.std(x))
    
    
    w = np.linalg.inv(A_train.T.dot(A_train) + np.exp(-18) * np.identity(M+1)).dot(A_train.T.dot(t_train))
    y_train = np.dot(A_train,w)
    z = (y_train-t_train).T.dot(y_train-t_train) + np.exp(-18)/2 * w.T.dot(w)
    zr = math.sqrt(z/len(x_train))
    
    
    # expressar nossa incerteza sobre o valor da variável alvo usando uma distribuição de probabilidade. 
    # Para esse propósito, assumiremos que, dado o valor de x, o valor correspondente de t possui uma distribuição 
    # gaussiana com uma média igual ao valor y (x, w) da curva polinomial - beta
    # Modelo de Regressão Bayesiano
    # w ~ N(w|0, lambd^(-1) * I)
    # y = A @ w
    # t ~ N(t|A @ w, beta^(-1))
    
    lambd = np.exp(-18) # 2e-3
    beta = 2 # parâmetro de precisão beta
    
    ndim = np.size(A_train, 1)
    mean_prev, precision_prev = np.zeros(ndim), lambd * np.eye(ndim)
    w_precision = precision_prev + beta * A_train.T @ A_train
    w_mean = np.linalg.solve( w_precision, precision_prev @ mean_prev + beta * A_train.T @ t_train )
    w_cov = np.linalg.inv(w_precision)
    
    # Calculando a média e o desvio padrão da distribuição preditiva (dp)
    # w_sample = np.random.multivariate_normal(w_mean, w_cov, size=sample_size ) # if sample_size is not None: - número de amostras a serem retiradas da distribuição preditiva
    # y_sample = A_test @ w_sample.T   # amostras da dp
    y = A_test @ w_mean # média da dp
    y_var = 1 / beta + np.sum(A_test @ w_cov * A_test, axis=1) 
    y_err = np.sqrt(y_var) # y_std = y_err  # desvio padrao da dp
    
    plt.figure(figsize=(8,5))
    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_correct, t_correct, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y, c="r", label="mean")
    plt.fill_between(x_test, y - y_err, y + y_err, color="pink", label="std.", alpha=0.5)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1.5, 1.5)
    plt.annotate("M=9", xy=(0.8, 1))
    plt.legend(bbox_to_anchor=(1.05, 1.), loc='best', borderaxespad=0.)
    
    
    #x_test = np.arange(-10, 10, 0.001)
    #y2 = norm.pdf(x_test,y,y_err)
    y2 = norm.pdf(w_mean,y,y_err)
    plt.plot(y2+0.5, x_test)
    #plt.fill_between(y2,x_test,0, alpha=0.3, color='b')
    
    
    # ============================Distribuição Normal=================================================
    # # Conjunto de objetos em uma cesta, a média é 8 e o desvio padrão é 2
    # # Qual a probabilidade de tirar um objeto que peso é menor que 6 quilos?
    # norm.cdf(6, 8, 2)
    # # Qual a probabilidade de tirar um objeto que o peso á maior que 6 quilos?
    # norm.sf(6, 8, 2)
    # 1 - norm.cdf(6, 8, 2)
    # # Qual a probabilidade de tirar um objeto que o peso é menor que 6 ou maior que 10 quilos?
    # norm.cdf(6, 8, 2) + norm.sf(10, 8, 2)
    # # Qual a probabilidade de tirar um objeto que o peso é menor que 10 e maior que 8 quilos?
    # norm.cdf(10, 8, 2) - norm.cdf(8, 8, 2)
    # dados = norm.rvs(size = 100) # GERA DADOS EM UMA DISTRIBUIÇÃO NORMAL
    # stats.probplot(dados, plot = plt)
    # stats.shapiro(dados)
    # =============================================================================