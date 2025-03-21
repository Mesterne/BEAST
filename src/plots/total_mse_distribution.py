import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_total_mse_distribution(
    total_mse_for_each_mts: np.array,  # Shape: (number of timeseries generated,)
) -> plt.Figure:
    """
    Returns a plot of the distribution of MSE values for all a list of MSE values.
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot histogram with density curve
    sns.histplot(
        total_mse_for_each_mts,
        bins=50,
        kde=True,
        alpha=0.7,
        edgecolor="black",
        ax=ax,
    )

    # Titles and labels
    ax.set_title("Distribution of Total MSE Values", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Squared Error", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    # Improve grid visibility
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Adjust layout for better spacing
    fig.tight_layout()

    return fig
