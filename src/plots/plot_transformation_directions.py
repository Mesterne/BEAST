import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


# Code partially from ChatGPT
def plot_transformation_directions(
    mts_dataset_pca: np.ndarray, transformation_indices: np.ndarray
) -> Figure:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    transformation_indices = np.array(transformation_indices)
    transformation_start: np.ndarray = mts_dataset_pca[transformation_indices[:, 0]]
    transformation_end: np.ndarray = mts_dataset_pca[transformation_indices[:, 1]]

    deltas = transformation_end - transformation_start

    sns.scatterplot(x=mts_dataset_pca[:, 0], y=mts_dataset_pca[:, 1], ax=ax, s=50)

    ax.quiver(
        transformation_start[:, 0],
        transformation_start[:, 1],
        deltas[:, 0],
        deltas[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        width=0.0015,
        alpha=0.25,
    )

    ax.set_title("Transformation Directions (Start → End)", fontsize=16)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.grid(True)

    return fig
