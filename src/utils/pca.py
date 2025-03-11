from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


class PCAWrapper:
    """
    A wrapper class for Principal Component Analysis (PCA).

    We created this to reduce the number of mistakes we could make.
    """

    def __init__(self, n_components: int = 2) -> None:
        """
        Initializes the PCA wrapper with the specified number of components.

        Args:
            n_components (int): The number of principal components to retain.
        """
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit_transform(self, mts_features: np.ndarray) -> np.ndarray:
        """
        Fits the PCA model to the features and transforms them into pca space.

        Args:
            mts_features (np.ndarray): Input features to be transformed.

        Returns:
            np.ndarray: The transformed features in the lower-dimensional space.
        """
        mts_pca_2d = self.pca.fit_transform(mts_features)
        self.is_fitted = True

        return mts_pca_2d

    def transform(self, mts_features: np.ndarray) -> np.ndarray:
        """
        Transforms the input features using a previously fitted PCA model.

        Args:
            mts_features (np.ndarray): Input features to be transformed.

        Returns:
            np.ndarray: The transformed features in the lower-dimensional space.

        Raises:
            RuntimeError: If the PCA model has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "The PCA model must be fitted using fit_transform() before calling transform()."
            )
        pca_components = self.pca.transform(mts_features)
        return pca_components
