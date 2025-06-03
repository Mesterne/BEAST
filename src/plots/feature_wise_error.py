import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from tqdm import tqdm

from src.data.constants import COLUMN_NAMES


# Plot code is partially from ChatGPT
def plot_distribution_of_feature_wise_error(mse_per_feature: np.array) -> Figure:
    num_features = mse_per_feature.shape[1]

    grid_size = math.ceil(math.sqrt(num_features))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(5 * grid_size, 4 * grid_size)
    )
    axes = axes.flatten()

    for i in tqdm(range(num_features)):
        sns.histplot(data=mse_per_feature[:, i], kde=True, color="blue", ax=axes[i])
        axes[i].set_title(f"Distribution of {COLUMN_NAMES[i]}")
        axes[i].set_xlabel("Error")
        axes[i].set_ylabel("Frequency")

    # Hide unused subplots
    for j in range(num_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Distribution of Differences for Prediction Features", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
