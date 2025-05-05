import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer

from src.data.constants import OUTPUT_DIR, PLOT_NAMES


def create_and_save_plots_of_ohe_activated_performances(
    ohe: np.ndarray, evaluation: np.ndarray, metric_name: str, dataset: str
):
    activated_indices: np.ndarray = np.argmax(ohe, axis=1)

    grouped_indices = [
        np.where(activated_indices == i)[0].tolist() for i in range(ohe.shape[1])
    ]

    grid_load_activated_evaluation = np.mean(evaluation[grouped_indices[0]], axis=0)
    grid_loss_activated_evaluation = np.mean(evaluation[grouped_indices[1]], axis=0)
    grid_temp_activated_evaluation = np.mean(evaluation[grouped_indices[2]], axis=0)
    total_evaluation = np.mean(evaluation, axis=0)

    df = pd.DataFrame(
        {
            "Activated OHE": ["Grid Load"] * len(grid_load_activated_evaluation)
            + ["Grid Loss"] * len(grid_loss_activated_evaluation)
            + ["Grid Temp"] * len(grid_temp_activated_evaluation)
            + ["In total, ignoring OHE"] * len(total_evaluation),
            "Feature": PLOT_NAMES + PLOT_NAMES + PLOT_NAMES + PLOT_NAMES,
            "metric": np.concatenate(
                [
                    grid_load_activated_evaluation,
                    grid_loss_activated_evaluation,
                    grid_temp_activated_evaluation,
                    total_evaluation,
                ]
            ),
        }
    )

    plot = plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Feature", y="metric", hue="Activated OHE", data=df)

    plt.xticks(rotation=45, ha="right")

    for container in ax.containers:
        if isinstance(container, BarContainer):
            labels = ax.bar_label(container, fmt="%.4f", label_type="edge", padding=3)
            for label in labels:
                label.set_rotation(90)

    plt.title(f"{dataset} {metric_name} for each feature based on OHE values")
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Feature")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)

    plt.tight_layout()

    plot.savefig(
        os.path.join(OUTPUT_DIR, f"{dataset}-{metric_name}-ohe-barplot.png"), dpi=600
    )

    grid_load_activated_evaluation = np.mean(grid_load_activated_evaluation)
    grid_loss_activated_evaluation = np.mean(grid_loss_activated_evaluation)
    grid_temp_activated_evaluation = np.mean(grid_temp_activated_evaluation)
    total_evaluation = np.mean(total_evaluation)

    df = pd.DataFrame(
        {
            "Activated OHE": ["Grid Load"] * len(grid_load_activated_evaluation)
            + ["Grid Loss"] * len(grid_loss_activated_evaluation)
            + ["Grid Temp"] * len(grid_temp_activated_evaluation)
            + ["In total, ignoring OHE"] * len(total_evaluation),
            "metric": np.concatenate(
                [
                    grid_load_activated_evaluation,
                    grid_loss_activated_evaluation,
                    grid_temp_activated_evaluation,
                    total_evaluation,
                ]
            ),
        }
    )

    plot = plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Activated OHE", y="metric", data=df)

    plt.xticks(rotation=45, ha="right")

    for container in ax.containers:
        if isinstance(container, BarContainer):
            labels = ax.bar_label(container, fmt="%.4f", label_type="edge", padding=3)
            for label in labels:
                label.set_rotation(90)

    plt.title(f"{dataset} {metric_name} based on OHE values")
    plt.ylabel(f"{metric_name}")
    plt.xlabel("OHE Activation")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)

    plt.tight_layout()

    plot.savefig(
        os.path.join(OUTPUT_DIR, f"{dataset}-{metric_name}-ohe-total-barplot.png"),
        dpi=600,
    )
