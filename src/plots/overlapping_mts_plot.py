import matplotlib.pyplot as plt
import numpy as np

from src.data.constants import BEATUIFUL_UTS_NAMES


def plot_overlapping_mts(set_of_mts: np.ndarray):
    num_mts = set_of_mts.shape[0]
    num_uts = set_of_mts.shape[1]
    num_time_steps = set_of_mts.shape[2]
    time_index = np.arange(num_time_steps)

    fig, axes = plt.subplots(nrows=1, ncols=num_uts, figsize=(6 * num_uts + 2, 6.5))

    if num_uts == 1:
        axes = [axes]

    for i, uts_name in enumerate(BEATUIFUL_UTS_NAMES):
        for mts in set_of_mts:
            (line_transformed,) = axes[i].plot(
                time_index, mts[i, :], color="red", linewidth=1
            )

        axes[i].set_title(f"{uts_name}")
        axes[i].set_xticks([])
        axes[i].xaxis.set_visible(False)

    plt.subplots_adjust(left=0.03, right=0.04)

    return fig
