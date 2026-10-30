"""Classification tree skeleton."""
from .base import BaseDecisionTree
from .node import Node
import numpy as np


class DecisionTreeClassifier(BaseDecisionTree):
    """Tree variant for classification.

    Parameters:
        max_depth (int or None): See ``BaseDecisionTree``.
        min_samples_split (int): See ``BaseDecisionTree``.
        criterion (str): Supported values: 'gini', 'entropy'.

    Each hook below documents what it receives and must return; see
    ``BaseDecisionTree`` for the full contract.
    """

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        super().__init__(max_depth, min_samples_split, criterion)

    def _build_tree(self, X, y, depth):
        """Recursively grow the tree.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at this node.
            y (ndarray, shape (n_samples,)): Targets at this node.
            depth (int): Current depth (root is 0).

        Returns:
            Node: Leaf (when a stopping rule triggers) or internal node with
            two subtrees built from the best split.
        """
        n = len(y)

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self._leaf_value(y))
        if n < self.min_samples_split:
            return Node(value=self._leaf_value(y))
        if self._impurity(y) == 0:
            return Node(value=self._leaf_value(y))

        split = self._best_split(X, y)
        if split is None:
            return Node(value=self._leaf_value(y))

        j, t = split
        left_mask = X[:, j] <= t
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_index=j, threshold=t, left=left, right=right)

    def _impurity(self, y):
        """Impurity of ``y`` per ``self.criterion``: Gini impurity 1 - sum(p_k^2) or entropy -sum(p_k * log2(p_k)), where p_k is the fraction of samples of class k.

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity (0 if pure).
        """
        n = len(y)
        if n == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        p_k = counts / n

        if self.criterion == "gini":
            return 1 - np.sum(p_k ** 2)
        if self.criterion == "entropy":
            p_k = p_k[p_k > 0]
            return -np.sum(p_k * np.log2(p_k))
        raise ValueError(f"Unsupported criterion: {self.criterion}")

    def _best_split(self, X, y):
        """Search all features/thresholds for the largest impurity decrease.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at the node.
            y (ndarray, shape (n_samples,)): Targets at the node.

        Returns:
            tuple (int, float) or None: ``(feature_index, threshold)``, or
            None if no split reduces impurity.
        """
        parent_imp = self._impurity(y)
        best_gain = 0
        best = None
        n = len(y)
        n_features = X.shape[1]

        for j in range(n_features):
            vals = np.unique(X[:, j])
            thresholds = (vals[:-1] + vals[1:]) / 2

            for t in thresholds:
                left_mask = X[:, j] <= t
                right_mask = ~left_mask
                n_l = np.sum(left_mask)
                n_r = np.sum(right_mask)
                if n_l == 0 or n_r == 0:
                    continue

                child_imp = (
                    n_l * self._impurity(y[left_mask])
                    + n_r * self._impurity(y[right_mask])
                ) / n
                gain = parent_imp - child_imp

                if gain > best_gain:
                    best_gain = gain
                    best = (j, t)

        if best_gain == 0:
            return None
        return best

    def _leaf_value(self, y):
        """Return the majority (most frequent) class of y.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The prediction for the leaf.
        """
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def _score(self, y_true, y_pred):
        """Metric for ``eval``: accuracy = fraction of correct predictions.

        Args:
            y_true (ndarray, shape (n_samples,)): True targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
        return np.mean(y_true == y_pred)
