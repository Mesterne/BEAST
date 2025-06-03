from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from src.data.constants import COLUMN_NAMES, UTS_NAMES
from src.utils.logging_config import logger
from src.utils.pca import PCAWrapper


def plot_pca_for_all_generated_mts(
    mts_dataset_features: np.ndarray,  # Shape
    mts_generated_features: np.ndarray,  # Shape = (num_generated_mts,num_uts_features)
    train_transformations: np.ndarray,
    evaluation_transformations: np.ndarray,  # Shape (,)
):
    fig = plt.figure(figsize=(12, 8))

    original_train_indices = np.unique(train_transformations[:, 1])
    evaluation_indices = np.unique(evaluation_transformations[:, 1])

    train_features = mts_dataset_features[original_train_indices]
    evaluation_features = mts_dataset_features[evaluation_indices]

    pca_transformer: PCAWrapper = PCAWrapper(n_components=2)

    dataset_pca: np.ndarray = pca_transformer.fit_transform(train_features)
    predicted_pca: np.ndarray = pca_transformer.transform(mts_generated_features)
    evaluation_set_pca: np.ndarray = pca_transformer.transform(evaluation_features)

    logger.info(
        f"Indices of generated MTS with PCA1 > 20: {np.where(predicted_pca[:, 0] > 20)[0]}"
    )

    ax = sns.scatterplot(
        x=dataset_pca[:, 0],
        y=dataset_pca[:, 1],
        label="Dataset",
        color="gray",
        alpha=0.7,
        s=50,
    )

    ax.scatter(
        *evaluation_set_pca.T,
        color="blue",
        s=75,
        label="Target MTS",
    )
    ax.scatter(*predicted_pca.T, color="red", s=100, label="Inferred MTS")

    ax.set_title("PCA Plot with Transformed MTS")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()

    fig.tight_layout()
    return fig


def plot_pca_for_all_generated_mts_for_each_uts(
    mts_dataset_features: np.ndarray,  # Shape
    mts_generated_features: np.ndarray,  # Shape = (num_generated_mts,num_uts_features)
    train_transformations: np.ndarray,
    evaluation_transformations: np.ndarray,  # Shape (,)
) -> Figure:

    original_train_indices = np.unique(train_transformations[:, 1])
    evaluation_indices = np.unique(evaluation_transformations[:, 1])

    num_uts: int = len(UTS_NAMES)
    fig, axes = plt.subplots(nrows=num_uts, ncols=1, figsize=(12, 8 * (num_uts)))

    if num_uts == 1:
        axes = [axes]

    ax = axes[0]

    for uts_name, ax in zip(UTS_NAMES, axes[0:]):
        uts_column_indices: List[int] = [
            j
            for j, col_name in enumerate(COLUMN_NAMES)
            if col_name.startswith(uts_name)
        ]

        uts_dataset_features: np.ndarray = mts_dataset_features[:, uts_column_indices]
        uts_dataset_train_features: np.ndarray = uts_dataset_features[
            original_train_indices
        ]
        uts_dataset_evaluation_features: np.ndarray = uts_dataset_features[
            evaluation_indices
        ]

        uts_predicted: np.ndarray = mts_generated_features[:, uts_column_indices]

        pca_transformer: PCAWrapper = PCAWrapper(n_components=2)
        uts_dataset_pca: np.ndarray = pca_transformer.fit_transform(
            uts_dataset_train_features
        )
        uts_evaluation_pca: np.ndarray = pca_transformer.transform(
            uts_dataset_evaluation_features
        )
        uts_predicted_pca: np.ndarray = pca_transformer.transform(uts_predicted)

        sns.scatterplot(
            x=uts_dataset_pca[:, 0],
            y=uts_dataset_pca[:, 1],
            label="Dataset",
            color="gray",
            alpha=0.7,
            s=50,
            ax=ax,
        )

        ax.scatter(
            *uts_evaluation_pca.T,
            color="blue",
            s=75,
            label="Target",
        )
        ax.scatter(
            *uts_predicted_pca.T,
            color="red",
            s=100,
            label="Predicted",
        )

        ax.set_title(f"PCA Plot with Transformed UTS for {uts_name}")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.legend()

    fig.tight_layout()
    return fig
