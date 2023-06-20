import numpy as np
from abc import ABC, abstractstaticmethod



class Metric(ABC):
    @abstractstaticmethod
    def name():
        pass

    @abstractstaticmethod
    def requires_probs():
        pass

    @abstractstaticmethod
    def calculate(y_pred: np.ndarray, y_true: np.ndarray):
        pass


class Accuracy(Metric):
    @staticmethod
    def name():
        return "accuracy"
    
    @staticmethod
    def requires_probs():
        return False

    @staticmethod
    def calculate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.sum(y_pred == y_true) / len(y_true)
    

class Precision(Metric):
    @staticmethod
    def name():
        return "precision"
    
    @staticmethod
    def requires_probs():
        return False
    
    @staticmethod
    def calculate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        false_positives = np.sum(np.logical_and(y_true == 0, y_pred == 1))
        
        precision = true_positives / (true_positives + false_positives)
        return precision


class Recall(Metric):
    @staticmethod
    def name():
        return "recall"
    
    @staticmethod
    def requires_probs():
        return False
    
    @staticmethod
    def calculate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        false_negatives = np.sum(np.logical_and(y_true == 1, y_pred == 0))
        
        recall = true_positives / (true_positives + false_negatives)
        return recall


class F1Score(Metric):
    @staticmethod
    def name():
        return "f1-score"
    
    @staticmethod
    def requires_probs():
        return False
    
    @staticmethod
    def calculate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        false_positives = np.sum(np.logical_and(y_true == 0, y_pred == 1))
        false_negatives = np.sum(np.logical_and(y_true == 1, y_pred == 0))
        
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


class AUC(Metric):
    @staticmethod
    def name():
        return "area under ROC curve"
    
    @staticmethod
    def requires_probs():
        return True
    
    @staticmethod
    def calculate(y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        # sort the predictions in descending order while keeping the corresponding true labels
        sorted_indices = np.argsort(y_pred_probs)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # total number of positive samples
        num_positives = np.sum(y_true == 1)
        
        # false positive rate (FPR) and true positive rate (TPR)
        tpr = np.cumsum(y_true_sorted) / num_positives
        fpr = np.cumsum(1 - y_true_sorted) / (len(y_true_sorted) - num_positives)
        
        # prepend 0 and append 1 to FPR and TPR - starting and ending points
        tpr = np.concatenate([[0], tpr, [1]])
        fpr = np.concatenate([[0], fpr, [1]])
        
        # area under roc
        auc = np.trapz(tpr, fpr)
        return auc
    

class LogLoss(Metric):
    @staticmethod
    def name():
        return "log loss"
    
    @staticmethod
    def requires_probs():
        return True
    
    @staticmethod
    def calculate(y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
        epsilon = 1e-15  # avoid division by zero
        y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)  # clip predicted probs
 
        loss = -np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))
        return loss