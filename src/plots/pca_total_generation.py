from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.data.constants import COLUMN_NAMES, UTS_NAMES
from src.utils.pca import PCAWrapper


def plot_pca_for_all_generated_mts(
    mts_features_train: np.ndarray,  # Shape = (num_mts, num_uts_features)
    mts_features_validation: np.ndarray,  # Shape = (num_mts, num_uts_features)
    mts_features_test: np.ndarray,  # Shape = (num_mts, num_uts_features)
    mts_generated_features: np.ndarray,  # Shape = (num_generated_mts,num_uts_features)
) -> Figure:

    num_uts: int = len(UTS_NAMES)
    fig, axes = plt.subplots(
        nrows=num_uts + 1, ncols=1, figsize=(12, 8 * (num_uts + 1))
    )

    if num_uts == 1:
        axes = [axes]

    ax = axes[0]
    pca_transformer: PCAWrapper = PCAWrapper(n_components=2)

    mts_features_all = np.vstack(
        [mts_features_train, mts_features_validation, mts_features_test]
    )

    mts_all_pca: np.ndarray = pca_transformer.fit_transform(mts_features_all)

    predicted_pca: np.ndarray = pca_transformer.transform(mts_generated_features)

    # Plot scatter points
    sns.scatterplot(
        x=mts_all_pca[:, 0],
        y=mts_all_pca[:, 1],
        label="Dataset",
        color="grey",
        s=50,
        ax=ax,
    )

    ax.scatter(*predicted_pca.T, color="orange", s=150, label="Predicted MTS")

    ax.set_title("PCA Plot with Transformed MTS")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()

    # Plot for each UTS
    for uts_name, ax in zip(UTS_NAMES, axes[1:]):
        uts_column_indices: List[int] = [
            j
            for j, col_name in enumerate(COLUMN_NAMES)
            if col_name.startswith(uts_name)
        ]

        uts_features_train: np.ndarray = mts_features_train[:, uts_column_indices]
        uts_features_validation: np.ndarray = mts_features_validation[
            :, uts_column_indices
        ]
        uts_features_test: np.ndarray = mts_features_test[:, uts_column_indices]
        uts_predicted: np.ndarray = mts_generated_features[:, uts_column_indices]

        pca_transformer: PCAWrapper = PCAWrapper(n_components=2)
        uts_train_pca: np.ndarray = pca_transformer.fit_transform(uts_features_train)
        uts_validation_pca: np.ndarray = pca_transformer.transform(
            uts_features_validation
        )
        uts_test_pca: np.ndarray = pca_transformer.transform(uts_features_test)

        uts_all_pca: np.ndarray = np.vstack(
            [uts_train_pca, uts_validation_pca, uts_test_pca]
        )
        uts_predicted_pca: np.ndarray = pca_transformer.transform(uts_predicted)

        sns.scatterplot(
            x=uts_all_pca[:, 0],
            y=uts_all_pca[:, 1],
            label="Dataset",
            color="grey",
            s=50,
            ax=ax,
        )

        ax.scatter(*uts_predicted_pca.T, color="orange", s=150, label="Predicted")

        ax.set_title(f"PCA Plot with Transformed UTS for {uts_name}")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.legend()

    plt.tight_layout()
    return fig
