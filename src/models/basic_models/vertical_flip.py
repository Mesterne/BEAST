import numpy as np


class VerticalFlip:
    def transform(self, X: np.ndarray) -> np.ndarray:
        max = np.max(X, axis=2, keepdims=True)
        flipped_X = max - X
        return flipped_X
