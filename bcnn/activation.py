import numpy as np
from abc import ABC, abstractmethod



class Activation(ABC):
    @abstractmethod
    def name():
        pass

    @abstractmethod
    def calculate(x: np.ndarray):
        pass

    @abstractmethod
    def calulate_derivative(x: np.ndarray):
        pass


class ReLU(Activation):
    def __init__(self):
        self.NAME = 'relu'

    def name(self):
        return self.NAME

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def calulate_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class Sigmoid(Activation):
    def __init__(self):
        self.NAME = 'sigmoid'

    def name(self):
        return self.NAME
    
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    def calulate_derivative(self, x: np.ndarray) -> np.ndarray:
        sig = Sigmoid.calculate(self, x)
        return sig * (1 - sig)
    

class TanH(Activation):
    def __init__(self):
        self.NAME = 'tanh'

    def name(self):
        return self.NAME
    
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self, x)
    
    def calulate_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(self, x) ** 2
    

class Softmax(Activation):
    def __init__(self):
        self.NAME = 'softmax'

    def name(self):
        return self.NAME
    
    def calculate(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(self, x - np.max(self, x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
    
    def calulate_derivative(self, x: np.ndarray) -> np.ndarray:
        softmax = Softmax.calculate(self, x)
        return softmax * (1 - softmax)
    

class Swish(Activation):
    def __init__(self):
        self.NAME = 'swish'

    def name(self):
        return self.NAME
    
    def calculate(self, x: np.ndarray) -> np.ndarray:
        return x * Sigmoid.calculate(self, x)
    
    def calulate_derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid = Softmax.calculate(self, x)
        return sigmoid + (1 - sigmoid) * x * sigmoid