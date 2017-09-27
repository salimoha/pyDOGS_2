from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier

import numpy as np


class SimpleMLP(BaseEstimator, ClassifierMixin):
    """Sinmple MLP with fixed hidden layer weights; ie original perceptron/Adaline
    Algo description: https://github.com/myazdani/ML-examples/blob/master/ADALINE.ipynb
    The simple MLP uses the SGDClassifier from Scikit learn and takes the same params.

    Parameters
    ----------
    num_neurons: int
        Number of hidden neurons to use. Defaults to 1000.

    gain: float
        Variance of the initialization weight for the hidden layer weights. Defaults to 0.01

    np_random_seed: int
        Seed for numpy random generator used for

    **kwargs:
        Additional arguments to send to SGDClassifier


    Attributes
    ----------
    clf_ : SGDClassifier object
        The SGDClassifier object used to fit

    weights_: array, shape (n_features, num_neurons)
        Weights that are initialized to inputs project to hidden layer

    Examples
    -------
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    Y = np.array([1, 1, 2, 2])
    clf = SimpleMLP(np_random_seed=111, alpha = .1)
    clf.fit(X,Y)
    print(clf.predict(X))

    """

    def __init__(self, num_neurons=1000, gain=0.01, np_random_seed=None, **kwargs):
        self.num_neurons = int(num_neurons)
        self.gain = gain
        self.np_random_seed = np_random_seed
        self.clf_ = SGDClassifier(**kwargs)

    def fit(self, X, y=None):
        # initialize weights
        np.random.seed(self.np_random_seed)
        self.weights_ = self.gain * np.random.randn(self.num_neurons, X.shape[1]).T
        # project
        Z = np.tanh(np.dot(X, self.weights_))
        # do actual "learning"
        self.clf_.fit(Z, y)

        return self

    def predict(self, X, y=None):
        Z = np.tanh(np.dot(X, self.weights_))
        return self.clf_.predict(Z)


def train_and_eval(**kwargs):
    mnist = fetch_mldata('MNIST original', data_home="mnist")
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.33, random_state=42)

    elm = SimpleMLP(**kwargs)

    elm.fit(X_train, y_train)
    y_pred = elm.predict(X_test)

    return accuracy_score(y_pred, y_test)


if __name__ == "__main__":
    accuracy = train_and_eval(np_random_seed=111, alpha=.1, random_state=2)
    print("The accuracy is:" + str(accuracy))