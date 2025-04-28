import numpy as np


class HorizontalFlip:
    def transform(self, X: np.ndarray) -> np.ndarray:
        flipped_X = np.flip(X, axis=2)
        return flipped_X
