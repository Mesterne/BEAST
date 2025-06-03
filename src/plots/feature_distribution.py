import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.constants import COLUMN_NAMES


# Code is partially from ChatGPT
def plot_feature_distribution(
    mts_feature_array: np.array,  # Shape (number_of_samples, number of features)
):
    num_features = len(COLUMN_NAMES)

    if mts_feature_array.shape[1] != num_features:
        raise ValueError("mts_feature_array shape does not match COLUMN_NAMES length.")

    grid_size = math.ceil(math.sqrt(num_features))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(6 * grid_size, 5 * grid_size)
    )
    axes = axes.flatten()

    feature_distribution = mts_feature_array

    for i in range(num_features):
        sns.histplot(
            data=feature_distribution[:, i],
            kde=True,
            ax=axes[i],
            alpha=0.7,
            edgecolor="black",
        )
        axes[i].set_title(f"{COLUMN_NAMES[i]}", fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Value", fontsize=12)
        axes[i].set_ylabel("Frequency", fontsize=12)
        axes[i].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Hide unused subplots
    for j in range(num_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Distribution of MTS Features", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
