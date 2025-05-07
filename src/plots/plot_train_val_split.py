import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.constants import OUTPUT_DIR


def plot_train_val_test_split(
    mts_dataset_pca: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    validation_pca = mts_dataset_pca[validation_indices, :]
    test_pca = mts_dataset_pca[test_indices, :]

    sns.scatterplot(
        x=mts_dataset_pca[:, 0],
        y=mts_dataset_pca[:, 1],
        label="Training set targets",
        color="grey",
        s=50,
        ax=ax,
    )

    ax.scatter(*validation_pca.T, color="green", s=50, label="Validation set targets")
    ax.scatter(*test_pca.T, color="red", s=50, label="Test set targets")

    ax.set_title("Distribution of train/validation/test splits")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "train_validation_test_split.png"))
    plt.close(fig)
    return fig
