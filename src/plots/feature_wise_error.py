import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math


def plot_distribution_of_feature_wise_error(differences_df: pd.DataFrame):
    """
    A plot to show the distribution of error over each distinct time series feature
    in an MTS.

    Parameters:
    differences_df (pd.DataFrame): DataFrame containing feature-wise errors.

    Returns:
    fig (matplotlib.figure.Figure): Matplotlib figure containing the plot.
    """
    num_features = len(differences_df.columns)

    # Determine grid size for the subplots
    grid_size = math.ceil(math.sqrt(num_features))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(5 * grid_size, 4 * grid_size)
    )
    axes = axes.flatten()

    for i, col in enumerate(differences_df.columns):
        sns.histplot(data=differences_df[col], kde=True, color="blue", ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel("Error")
        axes[i].set_ylabel("Frequency")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Distribution of Differences for Prediction Features", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
