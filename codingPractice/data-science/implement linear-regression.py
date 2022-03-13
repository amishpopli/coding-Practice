"""
Linear regression - Fit a straight line

line to fit
y = mx + b

cost function
mse = 1/n *(y - (wx+b))^2

w = w - lr*dw
b = b - lr*db

dw = 1/n*-2x(y-(mx+b))
db = i/n*(y - (mx+b))
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

X, y = datasets.make_regression(n_samples=1000, n_features=4, noise=20, random_state=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


class LinearRegressionScratch:
    def __init__(self, lr: float = 0.01, n_iters: int = 10):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            print(np.dot(X, (y_predicted -y)))
            dw = 2 * (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def accuracy_mse(self, X, y_true):
        return (1/X.shape[0])*np.sum(np.power((y_true - np.dot(X, self.weights) - self.bias), 2))


model = LinearRegressionScratch()

model.fit(X_train, y_train)

print(model.weights)

# print(model.predict(X_test))
#
# print(y_test)
#
# print(model.accuracy_mse(X_test, y_test))