"""
supervised learing method

we choose a n_neighbours variable

To classify a point , we use the above variable to find the closest neighbour
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import Counter

iris = datasets.load_iris()
X, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


class KnnScratch:
    def __init__(self, K: int = 3, distance: str = 'euclidean'):
        self.K = K
        self.distance = distance

    def fit(self, X: np.array, y: np.array) -> None:
        self.X_train = X
        self.y_train = y

    def predict(self, test_data: np.array) -> np.array:
        labels = [self._predict(row) for row in test_data]
        return np.array(labels)

    @staticmethod
    def accuracy(pred_labels: np.array, true_labels: np.array) -> float:
        return np.sum(np.equal(pred_labels, true_labels))/pred_labels.shape[0]

    def _predict(self, row: np.array) -> int:
        # find distance from each point
        distances = [self._EuclideanDistance(row, point) for point in self.X_train]

        # find top n_neighbours
        nearest_labels = self.y_train[np.argsort(distances)[0:self.K]]

        # majority vote
        label = Counter(nearest_labels).most_common()[0][0]

        return label

    @staticmethod
    def _EuclideanDistance(x: np.array, y: np.array) -> float:
        return np.sqrt(np.dot((x - y).T, (x - y)))


model = KnnScratch(K=2)
model.fit(x_train, y_train)

print(model.predict(x_test))

print(model.accuracy(model.predict(x_test), y_test))

print(model.predict(x_test) == y_test)