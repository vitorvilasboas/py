# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:11:05 2019
@author: Vitor Vilas-Boas
"""

import numpy as np
from scipy.stats import truncnorm
from contextlib import contextmanager



# prml.nn.config.py
class Config(object):
    __dtype = np.float32
    __is_updating_bn = False
    __available_dtypes = (np.float16, np.float32, np.float64)
    __enable_backprop = True

    @property
    def dtype(self):
        return self.__dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype in self.__available_dtypes:
            self.__dtype = dtype
        else:
            raise ValueError

    @property
    def is_updating_bn(self):
        return self.__is_updating_bn

    @is_updating_bn.setter
    def is_updating_bn(self, flag):
        assert(isinstance(flag, bool))
        self.__is_updating_bn = flag

    @property
    def enable_backprop(self):
        return self.__enable_backprop

    @enable_backprop.setter
    def enable_backprop(self, flag):
        assert(isinstance(flag, bool))
        self.__enable_backprop = flag


config = Config()

## prml.nn.queue.py
class BackPropQueue(object):

    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def enqueue(self, array):
        array.is_in_queue = True
        self.queue.append(array)

    def dequeue(self, depth_to_dequeue):
        queue = self.queue[0]
        for candidate in self.queue:
            if candidate.depth == depth_to_dequeue:
                queue = candidate
                break
            elif candidate.depth > queue.depth:
                queue = candidate
        self.queue.remove(queue)
        queue.is_in_queue = False
        return queue


backprop_queue = BackPropQueue()

## prml.nn.array.array.py
class Array(object):
    __array_ufunc__ = None

    def __init__(self, value, parent=None):
        self.value = np.atleast_1d(value)
        self.parent = parent
        self.grad = None
        self.gradtmp = None
        self.depth = 0 if parent is None else parent._out_depth()
        self.is_in_queue = False

    def __repr__(self):
        return f"Array(shape={self.value.shape}, dtype={self.value.dtype})"

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

    @property
    def dtype(self):
        return self.value.dtype

    def backward(self, delta=None):
        if delta is None:
            delta = np.ones_like(self.value).astype(config.dtype)
        assert(delta.shape == self.value.shape)
        self._backward(delta)
        backprop_queue.enqueue(self)
        depth = self.depth
        while(len(backprop_queue)):
            queue = backprop_queue.dequeue(depth)
            if queue.parent is not None:
                queue.parent.backward(queue.gradtmp)
            queue.update_grad(queue.gradtmp)
            queue.gradtmp = None
            depth = queue.depth

    def update_grad(self, grad):
        if self.grad is None:
            self.grad = np.copy(grad)
        else:
            self.grad += grad

    def cleargrad(self):
        self.grad = None
        self.gradtmp = None

    def _backward(self, delta):
        if delta is None:
            return
        assert(delta.shape == self.shape)
        if self.gradtmp is None:
            self.gradtmp = np.copy(delta)
        else:
            self.gradtmp += delta

    def __add__(self, arg):
        raise NotImplementedError

    def __radd__(self, arg):
        raise NotImplementedError

    def __truediv__(self, arg):
        raise NotImplementedError

    def __rtruediv__(self, arg):
        raise NotImplementedError

    def __matmul__(self, arg):
        raise NotImplementedError

    def __rmatmul__(self, arg):
        raise NotImplementedError

    def __mul__(self, arg):
        raise NotImplementedError

    def __rmul__(self, arg):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __pow__(self, arg):
        raise NotImplementedError

    def __rpow__(self, arg):
        raise NotImplementedError

    def __sub__(self, arg):
        raise NotImplementedError

    def __rsub__(self, arg):
        raise NotImplementedError

    def flatten(self):
        raise NotImplementedError

    def reshape(self, *args):
        raise NotImplementedError

    def swapaxes(self, *args):
        raise NotImplementedError

    def mean(self, axis=None, keepdims=False):
        raise NotImplementedError

    def prod(self):
        raise NotImplementedError

    def sum(self, axis=None, keepdims=False):
        raise NotImplementedError


def array(array_like):
    return Array(np.array(array_like, dtype=config.dtype))


def asarray(array_like):
    if isinstance(array_like, Array):
        return array_like
    return Array(np.asarray(array_like, dtype=config.dtype))



         
            
### prml.nn.function.py           
class Function(object):
    enable_auto_broadcast = False

    def forward(self, *args, **kwargs):
        self.args = [self._convert2array(arg) for arg in args]
        if self.enable_auto_broadcast:
            self.args = self._autobroadcast(self.args)
        self.kwargs = kwargs
        out = self._forward(*tuple(arg.value for arg in self.args), **kwargs)
        if config.enable_backprop:
            return Array(out, parent=self)
        else:
            return Array(out, parent=None)

    def backward(self, delta):
        dargs = self._backward(delta, *tuple(arg.value for arg in self.args), **self.kwargs)
        if isinstance(dargs, tuple):
            for arg, darg in zip(self.args, dargs):
                arg._backward(darg)
                if not arg.is_in_queue:
                    backprop_queue.enqueue(arg)
        else:
            self.args[0]._backward(dargs)
            if not self.args[0].is_in_queue:
                backprop_queue.enqueue(self.args[0])

    def _out_depth(self):
        return max([arg.depth for arg in self.args]) + 1

    @staticmethod
    def _autobroadcast(arg):
        return broadcast(arg)

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _backward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _convert2array(arg):
        if not isinstance(arg, Array):
            return asarray(arg)
        else:
            return arg


class BroadcastTo(Function):
    """
    Broadcast a tensor to an new shape
    """

    def __init__(self, shape):
        self.shape = shape

    def _forward(self, x):
        output = np.broadcast_to(x, self.shape)
        return output

    @staticmethod
    def _backward(delta, x):
        dx = delta
        xdim = getattr(x, "ndim", 0)
        xshape = getattr(x, "shape", ())
        if delta.ndim != xdim:
            dx = dx.sum(axis=tuple(range(dx.ndim - xdim)))
            if isinstance(dx, np.number):
                dx = np.array(dx)
        axis = tuple(i for i, len_ in enumerate(xshape) if len_ == 1)
        if axis:
            dx = dx.sum(axis=axis, keepdims=True)
        return dx


def broadcast_to(x, shape):
    """
    Broadcast a tensor to an new shape
    """
    return BroadcastTo(shape).forward(x)


def broadcast(args):
    """
    broadcast list of tensors to make them have the same shape

    Parameters
    ----------
    args : list
        list of Tensor to be aligned

    Returns
    -------
    list
        list of Tensor whose shapes are aligned
    """
    shape = np.broadcast(*(arg.value for arg in args)).shape
    for i, arg in enumerate(args):
        if arg.shape != shape:
            args[i] = BroadcastTo(shape).forward(arg)
    return args


### prml.nn.math.square.py 
class Square(Function):

    @staticmethod
    def _forward(x):
        return np.square(x)

    @staticmethod
    def _backward(delta, x):
        return 2 * delta * x


def square(x):
    return Square().forward(x)


### prml.nn.math.sum.py
class SumAxisOrKeepdims(Function):
    """
    summation along given axis
    y = sum_i=1^N x_i
    """

    def __init__(self, axis=None, keepdims=False):
        if isinstance(axis, int):
            axis = (axis,)
        self.axis = axis
        self.keepdims = keepdims

    def _forward(self, x):
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def _backward(self, delta, x):
        if isinstance(delta, np.ndarray) and (not self.keepdims) and (self.axis is not None):
            axis_positive = []
            for axis in self.axis:
                if axis < 0:
                    axis_positive.append(x.ndim + axis)
                else:
                    axis_positive.append(axis)
            for axis in sorted(axis_positive):
                delta = np.expand_dims(delta, axis)
        dx = np.broadcast_to(delta, x.shape)
        return dx

class SumSimple(Function):

    @staticmethod
    def _forward(x):
        return x.sum()

    @staticmethod
    def _backward(delta, x):
        return np.broadcast_to(delta, x.shape)


def sum(x, axis=None, keepdims=False):
    """
    returns summation of the elements along given axis
    y = sum_i=1^N x_i
    """
    x = Function._convert2array(x)
    if x.ndim == 1:
        return SumAxisOrKeepdims(axis=axis, keepdims=True).forward(x)
    elif axis is None and keepdims == False:
        return SumSimple().forward(x)
    return SumAxisOrKeepdims(axis=axis, keepdims=keepdims).forward(x)


### prml.nn.random.normal.py 
def normal(mean, std, size):
    return asarray(np.random.normal(mean, std, size))

def truncnormal(min, max, scale, size):
    return asarray(truncnorm(a=min, b=max, scale=scale).rvs(size))


### prml.nn.array.zeros.py 
def zeros(size):
    return Array(np.zeros(size, dtype=config.dtype))


### prml.nn.nonlinear.tanh.py 
class Tanh(Function):

    def _forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def _backward(self, delta, x):
        dx = delta * (1 - self.out ** 2)
        return dx


def tanh(x):
    return Tanh().forward(x)


### prml.nn.optimizer.optimizer.py
class Optimizer(object):

    def __init__(self, parameter: dict, learning_rate: float):
        if isinstance(parameter, list):
            self.parameter = {f"parameter{i}" : param for i, param in enumerate(parameter)}
        elif isinstance(parameter, dict):
            self.parameter = parameter
        self.learning_rate = learning_rate
        self.iter_count = 0

    def increment_iter_count(self):
        self.iter_count += 1

    def minimize(self, loss):

        self.learning_rate *= -1
        self.optimize(loss)
        self.iter_count += 1

    def maximize(self, score):
        if self.learning_rate < 0:
            self.learning_rate *= -1
        self.optimize(score)

    def optimize(self, array):
        self.increment_iter_count()
        array.backward()
        self.update()

    def update(self):
        raise NotImplementedError


### prml.nn.optimizer.adam.py
class Adam(Optimizer):
    """
    Adam optimizer
    initialization
    m1 = 0 (Initial 1st moment of gradient)
    m2 = 0 (Initial 2nd moment of gradient)
    n_iter = 0
    update rule
    n_iter += 1
    learning_rate *= sqrt(1 - beta2^n) / (1 - beta1^n)
    m1 = beta1 * m1 + (1 - beta1) * gradient
    m2 = beta2 * m2 + (1 - beta2) * gradient^2
    param += learning_rate * m1 / (sqrt(m2) + epsilon)
    """

    def __init__(self, parameter, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        construct Adam optimizer
        Parameters
        ----------
        parameters : dict
            dict of parameters to be optimized
        learning_rate : float
        beta1 : float
            exponential decay rate for the 1st moment
        beta2 : float
            exponential decay rate for the 2nd moment
        epsilon : float
            small constant to be added to denominator for numerical stability
        Attributes
        ----------
        n_iter : int
            number of iterations performed
        moment1 : dict
            1st moment of each learnable parameter
        moment2 : dict
            2nd moment of each learnable parameter
        """
        super().__init__(parameter, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moment1 = {}
        self.moment2 = {}
        for key, param in self.parameter.items():
            self.moment1[key] = np.zeros(param.shape, dtype=config.dtype)
            self.moment2[key] = np.zeros(param.shape, dtype=config.dtype)

    def update(self):
        """
        update parameter of the neural network
        """
        lr = (
            self.learning_rate
            * (1 - self.beta2 ** self.iter_count) ** 0.5
            / (1 - self.beta1 ** self.iter_count))
        for kp in self.parameter:
            p, m1, m2 = self.parameter[kp], self.moment1[kp], self.moment2[kp]
            if p.grad is None:
                continue
            m1 += (1 - self.beta1) * (p.grad - m1)
            m2 += (1 - self.beta2) * (p.grad ** 2 - m2)
            p.value += lr * m1 / (np.sqrt(m2) + self.epsilon)


### prml.nn.network.py
class Network(object):

    def __init__(self):
        self._setting_parameter = False
        self.parameter = {}

    @property
    def setting_parameter(self):
        return getattr(self, "_setting_parameter", False)

    @contextmanager
    def set_parameter(self):
        prev_scope = self._setting_parameter
        object.__setattr__(self, "_setting_parameter", True)
        try:
            yield
        finally:
            object.__setattr__(self, "_setting_parameter", prev_scope)

    def __setattr__(self, key, value):
        if self.setting_parameter:
            if isinstance(value, Array):
                self.parameter[self.__class__.__name__ + "." + key] = value
            elif isinstance(value, Network):
                for name, param in value.parameter.items():
                    self.parameter[self.__class__.__name__ + "." + key + "." + name] = param

        object.__setattr__(self, key, value)

    def clear(self):
        for param in self.parameter.values():
            param.cleargrad()

            
################# 5.1 Feed-forward Network Functions
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
            
np.random.seed(1234)

class RegressionNetwork(Network):
    
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        with self.set_parameter():
            self.w1 = truncnormal(-2, 2, 1, (n_input, n_hidden))
            self.b1 = zeros(n_hidden)
            self.w2 = truncnormal(-2, 2, 1, (n_hidden, n_output))
            self.b2 = zeros(n_output)
    
    def __call__(self, x):
        h = tanh(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

#def create_toy_data(func, n=50):
#    x = np.linspace(-1, 1, n)[:, None]
#    return x, func(x)

#def sinusoidal(x):
#    return np.sin(np.pi * x)

#def heaviside(x):
#    return 0.5 * (np.sign(x) + 1)

#if __name__ == "__main__":


n = 50
x_train = np.linspace(-1, 1, n)[:, None]
f_square = np.square(x_train)
f_sin = np.sin(np.pi * x_train)
f_abs = np.abs(x_train)
f_heaviside = 0.5 * (np.sign(x_train) + 1)

func_list = [f_square, f_sin, f_abs, f_heaviside]


plt.figure(figsize=(20, 10))
x = np.linspace(-1, 1, 1000)[:, None]


n_input = 1
n_hidden = 3
n_output = 1


for i, func, n_iter in zip(range(1, 5), func_list, [1000, 10000, 10000, 10000]):
    plt.subplot(2, 2, i)
    y_train = func
    #model = RegressionNetwork(1, 3, 1)
    
    w1 = asarray(truncnorm(a=-2, b=2, scale=1).rvs((n_input, n_hidden))).value #truncnorm(a=min, b=max, scale=scale).rvs(size)
    b1 = np.zeros(n_hidden, dtype=np.float32) # size = n_hidden
    w2 = asarray(truncnorm(a=-2, b=2, scale=1).rvs((n_input, n_hidden))).value 
    b2 = np.zeros(n_output, dtype=np.float32) # size = n_output
    
    learning_rate=0.1# 0.001
    beta1=0.9
    beta2=0.999
    epsilon=1e-8
    
    moment1 = {}
    moment2 = {}
    for key, param in parameter.items():
        moment1[key] = np.zeros(param.shape, dtype=np.float32)
        moment2[key] = np.zeros(param.shape, dtype=np.float32)
    
    
    
    #optimizer = Adam(model.parameter, 0.1)
    for _ in range(n_iter):
        h = tanh(x_train @ w1 + b1)
        loss = square(y_train - (h @ w2 + b2)).sum()
        

        optimizer.minimize(loss)
    
    h = tanh(x @ w1 + b1)
    y = h @ w2 + b2
    plt.scatter(x_train, y_train, s=10)
    plt.plot(x, y, color="r")
plt.show()