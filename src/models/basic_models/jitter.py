import numpy as np


class Jitter:
    def transform(self, X: np.ndarray):
        print("HER: ", X.shape)

        means = np.mean(X, axis=2, keepdims=True)
        std = np.std(X, axis=2, keepdims=True)

        jitter = np.random.uniform(low=-std, high=std, size=X.shape) * 0.75
        return X + jitter
