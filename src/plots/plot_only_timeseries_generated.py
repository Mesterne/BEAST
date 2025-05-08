import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from src.data.constants import BEATUIFUL_UTS_NAMES


def plot_only_timeseries_generated(
    original_mts: np.ndarray,
    target_mts: np.ndarray,
    transformed_mts: np.ndarray,
) -> Figure:
    num_uts = original_mts.shape[0]
    num_time_steps = original_mts.shape[1]
    time_index = np.arange(num_time_steps)

    fig, axes = plt.subplots(nrows=1, ncols=num_uts, figsize=(6 * num_uts + 2, 6.5))

    if num_uts == 1:
        axes = [axes]

    handles = []
    labels = []

    for i, uts_name in enumerate(BEATUIFUL_UTS_NAMES):
        # Use matplotlib directly to get proper handles
        (line_original,) = axes[i].plot(
            time_index, original_mts[i, :], color="grey", linestyle="--", linewidth=1
        )
        (line_target,) = axes[i].plot(
            time_index, target_mts[i, :], color="blue", linestyle="--", linewidth=1
        )
        (line_transformed,) = axes[i].plot(
            time_index, transformed_mts[i, :], color="red", linewidth=2
        )

        if i == 0:
            handles.extend([line_original, line_target, line_transformed])
            labels.extend(["Original", "Target", "Generated"])

        axes[i].set_title(f"{uts_name}")
        axes[i].set_xticks([])
        axes[i].xaxis.set_visible(False)

    plt.subplots_adjust(left=0.05, right=0.90)
    fig.legend(
        handles=handles,
        labels=labels,
        title="Legend",
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=13,
        title_fontsize=15,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        labelspacing=1.5,
        handlelength=3,
        handletextpad=1.0,
        borderaxespad=2.0,
        alignment="left",
    )

    return fig
