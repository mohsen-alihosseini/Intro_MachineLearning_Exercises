import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover - depends on local environment
    LogisticRegression = None


class LogisticRegressionClassifier:
    def __init__(self, max_iter=2000, random_state=0):
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        """
        Train a logistic-regression classifier on the given feature matrix.

        Requirements:
            - convert X and y to NumPy arrays
            - validate that X has shape (n_samples, n_features)
            - validate that y is one-dimensional
            - validate that X and y contain the same number of samples
            - create and fit sklearn.linear_model.LogisticRegression
            - return self
        """
        if LogisticRegression is None:
            raise ImportError("scikit-learn is required for LogisticRegressionClassifier.")
        X_train = np.asarray(X)
        y_train = np.asarray(y)
        if X_train.ndim != 2:
            raise ValueError("Shape of X_train does not match (n_samples, n_features)")
        if len(X_train) != len(y_train):
            raise ValueError("Length of X_train and y_train does not match")
        if y_train.ndim != 1:
            raise ValueError("y is not one-dimensional")
        self.model = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """
        Predict labels for one or more input samples.

        Requirements:
            - raise an error if fit() was not called first
            - accept either a single sample or a full batch
            - validate the feature dimension
            - return the model predictions as a NumPy array
        """
        if self.model is None:
            raise ValueError("LogisticRegressionClassifier has not been fitted yet.")
        X_train = np.asarray(X)
        if X_train.ndim == 1:
            X_train = X_train.reshape(1, -1)
        n_features = self.model.coef_.shape[1]
        if X_train.shape[1] != n_features:
            raise ValueError("Shape of X_train does not match n_features")
        return np.asarray(self.model.predict(X_train))
