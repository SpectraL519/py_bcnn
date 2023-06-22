import numpy as np
from abc import ABC, abstractmethod



class Normalizer(ABC):
    @abstractmethod
    def name():
        pass

    @abstractmethod
    def normalize(x: np.ndarray):
        pass


class L1(Normalizer):
    def __init__(self):
        self.NAME = 'l1'

    def name(self):
        return self.NAME
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, ord=1)


class L2(Normalizer):
    def __init__(self):
        self.NAME = 'l2'

    def name(self):
        return self.NAME
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, ord=2)
    

class ZScore(Normalizer):
    def __init__(self):
        self.NAME = 'z-score'

    def name(self):
        return self.NAME
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / std
    

class MinMax(Normalizer):
    def __init__(self):
        self.NAME = 'min-max'

    def name(self):
        return self.NAME
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        min_val = np.min(x)
        max_val = np.max(x)
        return (x - min_val) / (max_val - min_val)
    

class LogTransform(Normalizer):
    def __init__(self):
        self.NAME = 'log-transform'

    def name(self):
        return self.NAME
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)
    

class BoxCox(Normalizer):
    def __init__(self, lmbda: int = None):
        self.NAME = 'box-cox'
        self.LMBDA = lmbda

    def name(self):
        return self.NAME
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (
            np.log(x)
            if self.LMBDA is None
            else (np.power(x, self.LMBDA) - 1) / self.LMBDA
        )
    

class YeoJohnson(Normalizer):
    def __init__(self, lmbda: int = None):
        self.NAME = 'yeo-johnson'
        self.LMBDA = lmbda

    def name(self):
        return self.NAME
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        if np.all(x >= 0):
            return (
                np.log(x + 1)
                if self.LMBDA == 0
                else (np.power(x + 1, self.LMBDA) - 1) / self.LMBDA
            )

        return (
            -np.log(-x + 1)
            if self.LMBDA == 2
            else -((-x + 1) ** (2 - self.LMBDA) - 1) / (2 - self.LMBDA)
        )