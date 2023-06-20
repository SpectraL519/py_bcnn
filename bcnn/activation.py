import numpy as np
from abc import ABC, abstractstaticmethod



class Activation(ABC):
    @abstractstaticmethod
    def name():
        pass

    @abstractstaticmethod
    def calculate(x: np.ndarray):
        pass

    @abstractstaticmethod
    def calulate_derivative(x: np.ndarray):
        pass


class Relu(Activation):
    @staticmethod
    def name():
        return "relu"

    @staticmethod
    def calculate(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def calulate_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class Sigmoid(Activation):
    @staticmethod
    def name():
        return "sigmoid"

    @staticmethod
    def calculate(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def calulate_derivative(x: np.ndarray) -> np.ndarray:
        sig = Sigmoid.calculate(x)
        return sig * (1 - sig)
    

class TanH(Activation):
    @staticmethod
    def name():
        return "tanh"

    @staticmethod
    def calculate(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def calulate_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
    

class Softmax(Activation):
    @staticmethod
    def name():
        return "softmax"

    @staticmethod
    def calculate(x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
    
    @staticmethod
    def calulate_derivative(x: np.ndarray) -> np.ndarray:
        softmax = Softmax.calculate(x)
        return softmax * (1 - softmax)
    

class Swish(Activation):
    @staticmethod
    def name():
        return "swish"

    @staticmethod
    def calculate(x: np.ndarray) -> np.ndarray:
        return x * Sigmoid.calculate(x)
    
    @staticmethod
    def calulate_derivative(x: np.ndarray) -> np.ndarray:
        sigmoid = Softmax.calculate(x)
        return sigmoid + (1 - sigmoid) * x * sigmoid