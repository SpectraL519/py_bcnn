from typing import Union
import numpy as np

from bcnn.activation import Activation
from bcnn.metrics import Metric
from bcnn.losses import Loss
from bcnn.normalizers import Normalizer



class BinaryClassifier:
    def __init__(self, 
        input_size: int,
        neurons: Union[list[int], tuple[int, ...]], 
        activation: Activation,
        metric: Metric,
        loss: Loss,
        normalizer: Normalizer = None
    ):
        self.neurons = [
            input_size, # input layer
            *neurons, # hidden layers
            1 # output layer
        ]

        self.num_layers = len(self.neurons)
        self.num_hidden_layers = self.num_layers - 1
        assert(self.num_layers > 1)

        self.weights = [
            np.random.randn(y, x) 
            for x, y in zip(self.neurons[:-1], self.neurons[1:])
        ]
        self.biases = [np.zeros((y, 1)) for y in self.neurons[1:]]
        self.activations = [None] * self.num_hidden_layers
        self.weighted_sums = [None] * self.num_hidden_layers

        self.activation = activation
        self.metric = metric
        self.loss = loss
        self.normalizer = normalizer


    def summary(self):
        print('\n'.join([
            f"Number of layers: {self.num_layers}",
            f"Number of neurons per layer:",
            '\n'.join([f"\t{i + 1}: {n}" for i, n in enumerate(self.neurons)]),
            f"Activation: {self.activation.name()}",
            f"Metric: {self.metric.name()}",
            f"Loss: {self.loss.name()}",
            f"Normalization: {self.normalizer.name() if self.normalizer else 'none'}"
        ]))


    def _feed_forward(self, X: np.ndarray):
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            self.weighted_sums[i] = np.dot(w, (X if i == 0 else self.activations[i - 1])) + b
            self.activations[i] = self.activation.calculate(self.weighted_sums[i])

    
    def _back_propagate(self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        learning_rate: float
    ):
        m = X.shape[1]

        # derivative vectors
        d_biases = [None] * self.num_hidden_layers
        d_weights = [None] * self.num_hidden_layers
        d_weighted_sums = [None] * self.num_hidden_layers

        # output layer
        d_weighted_sums[-1] = self.activations[-1] - Y
        d_weights[-1] = np.dot(d_weighted_sums[-1], self.activations[-2].T) / m
        d_biases[-1] = np.sum(d_weighted_sums[-1], axis=1, keepdims=True) / m

        # hidden layers
        for i in reversed(range(0, self.num_hidden_layers - 1)):
            d_weighted_sums[i] = np.dot(
                self.weights[i + 1].T, 
                d_weighted_sums[i + 1]
            ) * self.activation.calulate_derivative(self.weighted_sums[i])
            d_weights[i] = np.dot(
                d_weighted_sums[i], 
                (X.T if i == 0 else self.activations[i - 1].T)
            ) / m
            d_biases[i] = np.sum(d_weighted_sums[i], axis=1, keepdims=True) / m

        for i in range(self.num_hidden_layers):
            self.weights[i] -= learning_rate * d_weights[i]
            self.biases[i] -= learning_rate * d_biases[i]


    def fit(self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        learning_rate: float = 0.001, 
        iterations: int = 1000,
        verbose: bool = True
    ):
        if self.normalizer:
            X = self.normalizer.normalize(X)

        step = iterations / 10
        for i in range(iterations):
            self._feed_forward(X)
            self._back_propagate(X, Y, learning_rate)
            if verbose and (i + 1) % step == 0:
                print(
                    f"iter {i + 1}: loss = {
                        round(self.loss.calculate(self.activations[-1], Y), 4)
                    }",
                    flush=True
                )


    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if (self.normalizer):
            X = self.normalizer.normalize(X)

        self._feed_forward(X)
        return np.where(self.activations[-1] >= threshold, 1, 0)
    

    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        if (self.normalizer):
            X = self.normalizer.normalize(X)

        self._feed_forward(X)
        return self.activations[-1]


    def evaluate(
        self, X: np.ndarray, Y: np.ndarray, threshold: float = 0.5
    ) -> tuple[float, float]:
        if (self.normalizer):
            X = self.normalizer.normalize(X)

        y_pred_probs = self.predict_probs(X)
        y_pred = np.where(self.activations[-1] >= threshold, 1, 0)

        loss = (
            self.loss.calculate(y_pred_probs, Y)
            if self.loss.from_probs()
            else self.loss.calculate(y_pred, Y)
        )
        metric = (
            self.metric.calculate(y_pred_probs, Y)
            if self.metric.from_probs()
            else self.metric.calculate(y_pred, Y)
        )
        return loss, metric
