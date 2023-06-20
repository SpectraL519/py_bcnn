import numpy as np
from abc import ABC, abstractstaticmethod



class Loss(ABC):
    @abstractstaticmethod
    def name():
        pass

    @abstractstaticmethod
    def requires_probs():
        pass

    @abstractstaticmethod
    def calculate(y_pred: np.ndarray, y_true: np.ndarray):
        pass


class MSE(Loss):
    @staticmethod
    def name():
        return "mean_sqared_error"
    
    @staticmethod
    def requires_probs():
        return False

    @staticmethod
    def calculate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.square(y_pred - y_true))
    

class MAE(Loss):
    @staticmethod
    def name():
        return "mean_absolute_error"
    
    @staticmethod
    def requires_probs():
        return False

    @staticmethod
    def calculate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.absolute(y_pred - y_true))
    

class BinaryCrossentropy(Loss):
    @staticmethod
    def name():
        return "binary_crossentropy"
    
    @staticmethod
    def requires_probs():
        return True
    
    @staticmethod
    def calculate(y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        epsilon = 1e-7  # avoid division by zero
        y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon) # avoid numerical instability
        
        # Calculate binary cross-entropy
        bce = -(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))
        return np.mean(bce)
    

class Hinge(Loss):
    @staticmethod
    def name():
        return "hinge_loss"
    
    @staticmethod
    def requires_probs():
        return True
    
    @staticmethod
    def calculate(y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        loss = np.maximum(0, 1 - y_true * y_pred_probs)
        return np.mean(loss)
    

class SquaredHinge(Loss):
    @staticmethod
    def name():
        return "squared_hinge_loss"
    
    @staticmethod
    def requires_probs():
        return True
    
    @staticmethod
    def calculate(y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        loss = np.exp(-y_true * y_pred_probs)
        return np.mean(loss)
    

class Exponential(Loss):
    @staticmethod
    def name():
        return "exponential_loss"
    
    @staticmethod
    def requires_probs():
        return True
    
    @staticmethod
    def calculate(y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        loss = np.maximum(0, 1 - y_true * y_pred_probs)
        return np.mean(np.square(loss))
    

class SigmoidCrossentropy(Loss):
    @staticmethod
    def name():
        return "sigmoid_crossentropy"
    
    @staticmethod
    def requires_probs():
        return True
    
    @staticmethod
    def calculate(y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        epsilon = 1e-7
        loss = -(y_true * np.log(y_pred_probs + epsilon) + (1 - y_true) * np.log(1 - y_pred_probs + epsilon))
        return np.mean(loss)
    

class Jaccard(Loss):
    @staticmethod
    def name():
        return "jaccard_loss"
    
    @staticmethod
    def requires_probs():
        return False
    
    @staticmethod
    def calculate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection

        epsilon = 1e-7
        jaccard = 1.0 - (intersection + epsilon) / (union + epsilon)
        return jaccard
    

class Dice(Loss):
    @staticmethod
    def name():
        return "jaccard_loss"
    
    @staticmethod
    def requires_probs():
        return False
    
    @staticmethod
    def calculate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)

        dice = 1.0 - (2.0 * intersection + 1e-7) / (union + 1e-7)
        return dice
