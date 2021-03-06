"""
Entropy = -Sum( p(x).log(p(x)))

Information Gain
IG = E(parent) - weighted avg.E(children)

what is leaf node - node with no child
"""

import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeScratch:
    def __init__(self, max_depth: int = 100, min_sample_split=2, n_features=None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        # traverse tree
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # greedy search
        best_feature, best_threshold = self._best_criteria(X, y, feat_idxs)
        left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        # generate_split
        left_idx, right_idx = self._split(X_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        n_l, n_r = len(left_idx), len(right_idx)
        entropy_left, entropy_right = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / (n_l + n_r)) * entropy_left + (n_r / (n_l + n_r)) * entropy_right
        return parent_entropy - child_entropy

    @staticmethod
    def _split(X_column, split_threshold):
        left_idx = np.argwhere(X_column <= split_threshold).flatten()
        right_idx = np.argwhere(X_column > split_threshold).flatten()
        return left_idx, right_idx

    @staticmethod
    def _most_common_label(y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    @staticmethod
    def _entropy(y):
        hist = np.bincount(y)
        px = hist / len(y)
        return -np.sum([p * np.log2(p) for p in px if p > 0])


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = DecisionTreeScratch(max_depth=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)
print("Accuracy:", acc)