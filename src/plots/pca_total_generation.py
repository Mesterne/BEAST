from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.data.constants import COLUMN_NAMES, UTS_NAMES
from src.utils.pca import PCAWrapper


def plot_pca_for_all_generated_mts(
    mts_dataset_features: np.ndarray,  # Shape
    mts_generated_features: np.ndarray,  # Shape = (num_generated_mts,num_uts_features)
    evaluation_set_indices: np.ndarray,  # Shape (,)
) -> Figure:

    num_uts: int = len(UTS_NAMES)
    fig, axes = plt.subplots(
        nrows=num_uts + 1, ncols=1, figsize=(12, 8 * (num_uts + 1))
    )

    if num_uts == 1:
        axes = [axes]

    ax = axes[0]
    pca_transformer: PCAWrapper = PCAWrapper(n_components=2)

    dataset_pca: np.ndarray = pca_transformer.fit_transform(mts_dataset_features)
    predicted_pca: np.ndarray = pca_transformer.transform(mts_generated_features)
    evaluation_set_pca: np.ndarray = dataset_pca[evaluation_set_indices, :]

    # Plot scatter points
    sns.scatterplot(
        x=dataset_pca[:, 0],
        y=dataset_pca[:, 1],
        label="Dataset",
        color="grey",
        s=50,
        ax=ax,
    )

    ax.scatter(*evaluation_set_pca.T, color="red", s=75, label="Evaluation set MTS")
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

        uts_dataset_features: np.ndarray = mts_dataset_features[:, uts_column_indices]
        uts_predicted: np.ndarray = mts_generated_features[:, uts_column_indices]

        pca_transformer: PCAWrapper = PCAWrapper(n_components=2)
        uts_dataset_pca: np.ndarray = pca_transformer.fit_transform(
            uts_dataset_features
        )
        uts_predicted_pca: np.ndarray = pca_transformer.transform(uts_predicted)

        sns.scatterplot(
            x=uts_dataset_pca[:, 0],
            y=uts_dataset_pca[:, 1],
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
