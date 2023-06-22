import numpy as np
from abc import ABC, abstractmethod



class Loss(ABC):
    @abstractmethod
    def name():
        pass

    @abstractmethod
    def from_probs():
        pass

    @abstractmethod
    def calculate(y_pred: np.ndarray, y_true: np.ndarray):
        pass


class MSE(Loss):
    def __init__(self):
        self.NAME = 'mean_sqared_error'
        self.FROM_PROBS = False

    def name(self):
        return self.NAME
    
    def from_probs(self):
        return self.FROM_PROBS
    
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.square(y_pred - y_true))
    

class MAE(Loss):
    def __init__(self):
        self.NAME = 'mean_absolute_error'
        self.FROM_PROBS = False

    def name(self):
        return self.NAME
    
    def from_probs(self):
        return self.FROM_PROBS
    
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.absolute(y_pred - y_true))
    

class BinaryCrossentropy(Loss):
    def __init__(self):
        self.NAME = 'binary_crossentropy'
        self.FROM_PROBS = True

    def name(self):
        return self.NAME
    
    def from_probs(self):
        return self.FROM_PROBS
    
    def calculate(self, y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        epsilon = 1e-7  # avoid division by zero
        y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon) # avoid numerical instability
        
        # Calculate binary cross-entropy
        bce = -(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))
        return np.mean(bce)
    

class Hinge(Loss):
    def __init__(self):
        self.NAME = 'hinge_loss'
        self.FROM_PROBS = True

    def name(self):
        return self.NAME
    
    def from_probs(self):
        return self.FROM_PROBS
    
    def calculate(self, y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        loss = np.maximum(0, 1 - y_true * y_pred_probs)
        return np.mean(loss)
    

class SquaredHinge(Loss):
    def __init__(self):
        self.NAME = 'squared_hinge_loss'
        self.FROM_PROBS = True

    def name(self):
        return self.NAME
    
    def from_probs(self):
        return self.FROM_PROBS
    
    def calculate(self, y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        loss = np.exp(-y_true * y_pred_probs)
        return np.mean(loss)
    

class Exponential(Loss):
    def __init__(self):
        self.NAME = 'exponential_loss'
        self.FROM_PROBS = True

    def name(self):
        return self.NAME
    
    def from_probs(self):
        return self.FROM_PROBS
    
    def calculate(self, y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        loss = np.maximum(0, 1 - y_true * y_pred_probs)
        return np.mean(np.square(loss))
    

class SigmoidCrossentropy(Loss):
    def __init__(self):
        self.NAME = 'sigmoid_crossentropy'
        self.FROM_PROBS = True

    def name(self):
        return self.NAME
    
    def from_probs(self):
        return self.FROM_PROBS
    
    def calculate(self, y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        epsilon = 1e-7
        loss = -(y_true * np.log(y_pred_probs + epsilon) + (1 - y_true) * np.log(1 - y_pred_probs + epsilon))
        return np.mean(loss)
    

class Jaccard(Loss):
    def __init__(self):
        self.NAME = 'jaccard_loss'
        self.FROM_PROBS = False

    def name(self):
        return self.NAME
    
    def from_probs(self):
        return self.FROM_PROBS
    
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection

        epsilon = 1e-7
        jaccard = 1.0 - (intersection + epsilon) / (union + epsilon)
        return jaccard
    

class Dice(Loss):
    def __init__(self):
        self.NAME = 'dice_loss'
        self.FROM_PROBS = False

    def name(self):
        return self.NAME
    
    def from_probs(self):
        return self.FROM_PROBS
    
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)

        dice = 1.0 - (2.0 * intersection + 1e-7) / (union + 1e-7)
        return dice
