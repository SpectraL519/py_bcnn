import numpy as np
from abc import ABC, abstractstaticmethod



class Normalizer(ABC):
    @abstractstaticmethod
    def name():
        pass

    @abstractstaticmethod
    def normalize(x: np.ndarray):
        pass


class L1(Normalizer):
    @staticmethod
    def name():
        return "L1"

    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, ord=1)


class L2(Normalizer):
    @staticmethod
    def name():
        return "L2"

    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, ord=2)
    

class ZScore(Normalizer):
    @staticmethod
    def name():
        return "z-score"
    
    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        mean = np.mean(x)
        std = np.std(x)
        return (x - mean) / std
    

class MinMax(Normalizer):
    @staticmethod
    def name():
        return "min-max"
    
    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        min_val = np.min(x)
        max_val = np.max(x)
        return (x - min_val) / (max_val - min_val)
    

class LogTransform(Normalizer):
    @staticmethod
    def name():
        return "log-transformation"
    
    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        return np.log(x)
    

class BoxCox(Normalizer):
    @staticmethod
    def name():
        return "box-cox"
    
    @staticmethod
    def normalize(x: np.ndarray, lmbda: int = None) -> np.ndarray:
        return (
            np.log(x)
            if lmbda is None
            else (np.power(x, lmbda) - 1) / lmbda
        )
    

class YeoJohnson(Normalizer):
    @staticmethod
    def name():
        return "yeo-johnson"
    
    @staticmethod
    def normalize(x: np.ndarray, lmbda: int = None) -> np.ndarray:
        if np.all(x >= 0):
            return (
                np.log(x + 1)
                if lmbda == 0
                else (np.power(x + 1, lmbda) - 1) / lmbda
            )

        return (
            -np.log(-x + 1)
            if lmbda == 2
            else -((-x + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
        )