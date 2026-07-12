import numpy as np


class NBNNClassifier:
    def __init__(self, metric="euclidean"):
        # Distance metric: "euclidean" or "cosine".
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Store training data and labels as NumPy arrays.

        Requirements:
            - convert X and y to NumPy arrays
            - validate shapes
            - store the sorted unique class labels in self.classes_
            - return self
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        if self.X_train.shape != (len(X), len(X[0])):
            raise ValueError("Shape of X_train does not match (n_samples, n_features)")
        if len(self.X_train) != len(self.y_train):
            raise ValueError("Length of X_train and y_train does not match")
        self.classes_ = np.unique(self.y_train)
        return self

    def _euclidean_distances(self, x):
        """Return the Euclidean distance from x to all training samples."""
        distance = np.sqrt(np.sum((self.X_train[:, None, :] - x[None, :, :])**2, axis=2))
        return distance

    def _cosine_distances(self, x):
        """
        Return the cosine distance from x to all training samples.

        Use the same convention as in knn.py:
            cosine_distance = 1 - cosine_similarity
        """
        eps = 1e-12
        mtrx = self.X_train @ x.T

        X_norm = np.sqrt(np.sum(self.X_train**2, axis=1))
        x_norm = np.sqrt(np.sum(x**2, axis=1))
        denom = X_norm[:, None] * x_norm[None, :]
        cosine_similarity = np.zeros_like(mtrx, dtype=float)
        valid = denom > eps
        cosine_similarity[valid] = mtrx[valid] / denom[valid]
        return 1.0 - cosine_similarity

    def _class_scores(self, distances):
        """
        Compute one score per class.

        For each class, use the distance of the nearest training sample from
        that class. The predicted class is the class with the smallest score.
        """
        n_samples = distances.shape[0]
        scores = np.full((n_samples, self.classes_.size), np.inf, dtype=float)

        for i, cls in enumerate(self.classes_):
            mask = (self.y_train == cls)
            if np.any(mask):
                # nearest training sample distance for this class
                scores[:, i] = np.min(distances[mask, :], axis=0)
        return scores

    def predict(self, X):
        """
        Predict labels for one or more samples with the NBNN rule.

        Requirements:
            - allow either a single sample or a batch
            - compute distances to all training samples
            - convert them into class-wise scores
            - return the class label with the smallest score
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.metric == "euclidean":
            # Compute Euclidean distances from x to all training samples.
            distance = self._euclidean_distances(X)
        elif self.metric == "cosine":
            # Compute cosine distances from x to all training samples.
            distance = self._cosine_distances(X)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        scores = self._class_scores(distance)
        class_indices = np.argmin(scores, axis=1)
        return self.y_train[class_indices]