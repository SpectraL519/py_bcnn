# py_bcnn
A simple feed forward binary classification neural network

<br />

### Environment setup

**Requires: python 3.9**

> Note: There hasn't yet been released an official numpy wheel for python 3.10

```
python -m venv env
./env/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --name bcnn
```

<br />
<br />
<br />

## Docs

### BinaryClassifier

Binary classification dense feed forward network

```
class bcnn.network.BinaryClassifier(input_size, neurons, activation, metric, loss, normalizer)
```

<br />

**Parameters:**

* `input_size : int` - size of the input vector
* `neurons : list[int] or tuple[int, ...]` - list of hidden layers neuron counts
* `activation : bcnn.Activation` - the activation function of the network
* `metric : bcnn.Metric` - the metric used to evalueate the model
* `loss : bcnn.Loss` - the loss function used to evaluate the model
* `normalizer : bcnn.Normalizer` - the normalization function applied to the data before trainig or evaluating the model or before making predictions

<br />

### Activation

`bcnn.activation.py` - A collection of activation functions:

* ReLU
* Sigmoid
* TanH
* Softmax
* Swish

> Each activation function is represented as a class deriving from abstract class `bcnn.activation.Activation`

<br />

### Metrics

`bcnn.metrics.py` - A collection of metric functions:

* Accuracy
* Precision
* Recall
* F1Score
* AUC
* LogLoss

> Each metric is represented as a class deriving from abstract class `bcnn.metrics.Metric`

<br />

### Losses

`bcnn.losses.py` - A collection of loss functions:

* MSE
* MAE
* BinaryCrossentropy
* Hinge
* SquaredHinge
* SigmoidCrossentropy
* Jaccard
* Dice

> Each loss function is represented as a class deriving from abstract class `bcnn.losses.Loss`

<br />

### Normalizers

`bcnn.normalizers.py` - A collection of activation functions:

* L1
* L2
* ZScore
* MinMax
* LogTransform
* BoxCox (Parameters: `lmbda : int, default = None`) 
* YeoJohnson (Parameters: `lmbda : int, default = 0`)

> Each normalizer is represented as a class deriving from abstract class `bcnn.normalizers.Normalizer`

<br />
<br />
<br />


### TODO
Refactor $L_1$ and $L_2$ normalizers - Create a single class $L_k(ord)$ which will support integer or None values of $k$ (if None normalizer becomes $L_\infty$)