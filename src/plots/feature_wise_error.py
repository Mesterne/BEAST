import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import math

from src.data.constants import COLUMN_NAMES


def plot_distribution_of_feature_wise_error(mse_per_feature: np.array) -> Figure:
    """
    A plot to show the distribution of error over each distinct time series feature
    in an MTS.

    Parameters:
    mse_per_feature (np.array): A numpy array of shape (num_timeseries, 12),
                                where each column represents the MSE of a specific feature.
    column_names (list[str]): A list of feature names corresponding to each column.

    Returns:
    fig (matplotlib.figure.Figure): Matplotlib figure containing the plot.
    """
    num_features = mse_per_feature.shape[1]

    # Determine global min and max for x-axis
    global_min = mse_per_feature.min()
    global_max = mse_per_feature.max()

    # Determine grid size for the subplots
    grid_size = math.ceil(math.sqrt(num_features))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(5 * grid_size, 4 * grid_size)
    )
    axes = axes.flatten()

    for i in range(num_features):
        sns.histplot(data=mse_per_feature[:, i], kde=True, color="blue", ax=axes[i])
        axes[i].set_title(f"Distribution of {COLUMN_NAMES[i]}")
        axes[i].set_xlabel("Error")
        axes[i].set_ylabel("Frequency")

        # Set consistent x-axis limits
        axes[i].set_xlim(global_min, global_max)

    # Hide unused subplots
    for j in range(num_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Distribution of Differences for Prediction Features", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
