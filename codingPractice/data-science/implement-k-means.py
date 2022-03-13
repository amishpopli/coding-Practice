"""
K means - We need to split data in k groups of similarity
 To calculate the similarity, we use eucledian distance

1. Initialize k points (called means randomly)
2. Categorize each item to the closest mean and update the means coordinates which are averages of items categorized so
    far
3. Repeat process for a given number of iterations

Assumptions:
1. Data is correct
"""

import pandas as pd
import numpy as np

data = pd.read_csv('data.csv', header=None)
data.columns = ['v' + str(i) for i in data.columns.tolist()]
train = data[[col for col in data.columns if col != 'v4']]
train = train.astype(float)


class KMeanScratch:
    def __init__(self, training_data: pd.DataFrame, K: int, distance: str = 'euclidean', iterations: int = 100):
        self.training_data = np.array(training_data)
        self.iterations = iterations
        self.distance = distance
        self.num_features = self.training_data.shape[1]
        self.num_points = self.training_data.shape[0]
        self.K = K
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = np.zeros((self.K, self.num_features))

    def __str__(self):
        print("The kmeans class is initialized")

    def fit(self):
        self._InitializeMeans()

        for _ in range(self.iterations):
            # classify in clusters
            self.clusters = self._create_clusters()
            # save old centroid
            old_centroid = self.centroids
            # update the centroid
            self.centroids = self._update_centroids()
            # check if converged
            if self._is_converged(old_centroid):
                break
            else:
                continue
        return self._get_labels()

    def predict(self, test_data: pd.DataFrame):
        if self.centroids == np.zeros((self.K, self.num_features)):
            raise Exception("please use the fit method to create a model")
        else:
            pred_clusters = [[] for _ in range(self.K)]

            for idx, row in test_data:
                cluster_idx = self._closest_centroid(row)
                pred_clusters[cluster_idx].append(idx)

            return self._get_labels(pred_clusters)

    def fit_predict(self):
        pass

    def _InitializeMeans(self):
        random_idx = np.random.choice(self.num_points, self.K, replace=False)
        for i, idx in enumerate(random_idx):
            self.centroids[i] = self.training_data[idx]

    @staticmethod
    def _EuclideanDistance(x: np.array, y: np.array):
        return np.sqrt(np.dot((x - y).T, (x - y)))

    def _closest_centroid(self, row: np.array):
        distances = [self._EuclideanDistance(row, centroid) for centroid in self.centroids]
        return np.array(distances).argmin()

    def _create_clusters(self, some_data: pd.DataFrame = None):
        clusters = [[] for _ in range(self.K)]
        for idx, row in enumerate(self.training_data if some_data is None else some_data):
            centroid_idx = self._closest_centroid(row)
            clusters[centroid_idx].append(idx)
        return clusters

    def _update_centroids(self):
        new_centroid = np.zeros((self.K, self.num_features))
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_means = np.mean(self.training_data[cluster], axis=0)
            new_centroid[cluster_idx] = cluster_means
        return new_centroid

    def _is_converged(self, old_centroid: np.array):
        distances = [self._EuclideanDistance(old_centroid[i], self.centroids[i]) for i in range(self.K)]
        return np.sum(distances) == 0

    def _get_labels(self, some_cluster=None):
        labels = np.empty(self.num_points)
        for cluster_idx, cluster in enumerate(self.clusters if some_cluster is None else some_cluster):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels


model = KMeanScratch(training_data=train, K=3)

labels = model.fit()

print(labels)