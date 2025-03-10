from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from src.utils.pca import PCAWrapper
from src.data.constants import COLUMN_NAMES, UTS_NAMES


def plot_pca_for_each_uts_with_transformed(
    mts_features_train: np.ndarray,  # Shape = (num_mts, num_uts_features)
    mts_features_validation: np.ndarray,  # Shape = (num_mts, num_uts_features)
    mts_features_test: np.ndarray,  # Shape = (num_mts, num_uts_features)
    original_mts_features: np.ndarray,  # Shape = (num_uts_features,)
    target_mts_features: np.ndarray,  # Shape = (num_uts_features,)
    predicted_mts_features: np.ndarray,  # Shape = (num_uts_features,)
) -> Figure:

    num_uts: int = len(UTS_NAMES)
    fig, axes = plt.subplots(
        nrows=num_uts + 1, ncols=1, figsize=(12, 8 * (num_uts + 1))
    )

    if num_uts == 1:
        axes = [axes]

    ax = axes[0]
    pca_transformer: PCAWrapper = PCAWrapper(n_components=2)

    # Reshape single samples to 2D. New shape (1, num_uts_features)
    original_mts_features = original_mts_features.reshape(1, -1)
    target_mts_features = target_mts_features.reshape(1, -1)
    predicted_mts_features = predicted_mts_features.reshape(1, -1)

    train_pca: np.ndarray = pca_transformer.fit_transform(mts_features_train)
    validation_pca: np.ndarray = pca_transformer.transform(mts_features_validation)
    test_pca: np.ndarray = pca_transformer.transform(mts_features_test)

    # We add all data points to one array
    mts_all_pca: np.ndarray = np.vstack([train_pca, validation_pca, test_pca])

    original_pca: np.ndarray = pca_transformer.transform(original_mts_features)
    target_pca: np.ndarray = pca_transformer.transform(target_mts_features)
    predicted_pca: np.ndarray = pca_transformer.transform(predicted_mts_features)

    # Plot scatter points
    sns.scatterplot(
        x=mts_all_pca[:, 0],
        y=mts_all_pca[:, 1],
        label="Dataset",
        color="grey",
        s=50,
        ax=ax,
    )
    ax.scatter(
        x=validation_pca[:, 0],
        y=validation_pca[:, 1],
        label="Target (Validation set)",
        color="red",
        s=50,
    )

    ax.scatter(*original_pca.T, color="blue", s=150, label="Original MTS")
    ax.scatter(*target_pca.T, color="green", s=150, label="Target MTS")
    ax.scatter(*predicted_pca.T, color="orange", s=150, label="Predicted MTS")

    # Draw dotted arrow from Original to Predicted
    ax.annotate(
        "",
        xy=predicted_pca.flatten(),
        xytext=original_pca.flatten(),
        arrowprops=dict(arrowstyle="->", linestyle="dotted", color="black", lw=2),
    )

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
        uts_original: np.ndarray = original_mts_features[:, uts_column_indices]
        uts_target: np.ndarray = target_mts_features[:, uts_column_indices]
        uts_predicted: np.ndarray = predicted_mts_features[:, uts_column_indices]

        pca_transformer: PCAWrapper = PCAWrapper(n_components=2)
        uts_train_pca: np.ndarray = pca_transformer.fit_transform(uts_features_train)
        uts_validation_pca: np.ndarray = pca_transformer.transform(
            uts_features_validation
        )
        uts_test_pca: np.ndarray = pca_transformer.transform(uts_features_test)

        uts_all_pca: np.ndarray = np.vstack(
            [uts_train_pca, uts_validation_pca, uts_test_pca]
        )
        uts_original_pca: np.ndarray = pca_transformer.transform(uts_original)
        uts_target_pca: np.ndarray = pca_transformer.transform(uts_target)
        uts_predicted_pca: np.ndarray = pca_transformer.transform(uts_predicted)

        sns.scatterplot(
            x=uts_all_pca[:, 0],
            y=uts_all_pca[:, 1],
            label="Dataset",
            color="grey",
            s=50,
            ax=ax,
        )
        ax.scatter(
            x=uts_validation_pca[:, 0],
            y=uts_validation_pca[:, 1],
            label="Target (Validation set)",
            color="red",
            s=50,
        )

        ax.scatter(*uts_original_pca.T, color="blue", s=150, label="Original")
        ax.scatter(*uts_target_pca.T, color="green", s=150, label="Target")
        ax.scatter(*uts_predicted_pca.T, color="orange", s=150, label="Predicted")

        # Draw dotted arrow from Original to Predicted
        ax.annotate(
            "",
            xy=uts_predicted_pca.flatten(),
            xytext=uts_original_pca.flatten(),
            arrowprops=dict(arrowstyle="->", linestyle="dotted", color="black", lw=2),
        )

        ax.set_title(f"PCA Plot with Transformed UTS for {uts_name}")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.legend()

    plt.tight_layout()
    return fig
