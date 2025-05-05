import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer

from src.data.constants import OUTPUT_DIR, PLOT_NAMES


def create_and_save_plots_of_ohe_activated_performances_feature_space(
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

    plt.title(
        f"FEATURE SPACE - {dataset} {metric_name} for each feature based on OHE values"
    )
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Feature")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)

    plt.tight_layout()

    plot.savefig(
        os.path.join(
            OUTPUT_DIR, f"FEATURE_SPACE_OHE {dataset}-{metric_name}-ohe-barplot.png"
        ),
        dpi=600,
    )
    plt.close(plot)

    grid_load_activated_evaluation = np.mean(grid_load_activated_evaluation)
    grid_loss_activated_evaluation = np.mean(grid_loss_activated_evaluation)
    grid_temp_activated_evaluation = np.mean(grid_temp_activated_evaluation)
    total_evaluation = np.mean(total_evaluation)

    df = pd.DataFrame(
        {
            "Activated OHE": ["Grid Load"]
            + ["Grid Loss"]
            + ["Grid Temp"]
            + ["In total, ignoring OHE"],
            "metric": [
                grid_load_activated_evaluation,
                grid_loss_activated_evaluation,
                grid_temp_activated_evaluation,
                total_evaluation,
            ],
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

    plt.title(f"FEATURE SPACE {dataset} {metric_name} based on OHE values")
    plt.ylabel(f"{metric_name}")
    plt.xlabel("OHE Activation")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)

    plt.tight_layout()

    plot.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"FEATURE_SPACE_OHE {dataset}-{metric_name}-ohe-total-barplot.png",
        ),
        dpi=600,
    )
    plt.close(plot)


def create_and_save_plots_of_ohe_activated_performances_forecasting_space(
    ohe: np.ndarray,
    train_metrics: np.ndarray,
    val_metrics: np.ndarray,
    test_metrics: np.ndarray,
    metric_name: str,
    retrain_on: str,
):
    delta_train = train_metrics[0] - train_metrics[1]
    delta_validation = val_metrics[0] - val_metrics[1]
    delta_test = test_metrics[0] - test_metrics[1]
    activated_indices: np.ndarray = np.argmax(ohe, axis=1)

    grouped_indices = [
        np.where(activated_indices == i)[0].tolist() for i in range(ohe.shape[1])
    ]

    train_grid_load_activated_delta = np.mean(delta_train[grouped_indices[0]], axis=0)
    train_grid_loss_activated_delta = np.mean(delta_train[grouped_indices[1]], axis=0)
    train_grid_temp_activated_delta = np.mean(delta_train[grouped_indices[2]], axis=0)
    train_total_evaluation = np.mean(delta_train, axis=0)

    validation_grid_load_activated_delta = np.mean(
        delta_validation[grouped_indices[0]], axis=0
    )
    validation_grid_loss_activated_delta = np.mean(
        delta_validation[grouped_indices[1]], axis=0
    )
    validation_grid_temp_activated_delta = np.mean(
        delta_validation[grouped_indices[2]], axis=0
    )
    validation_total_evaluation = np.mean(delta_validation, axis=0)

    test_grid_load_activated_delta = np.mean(delta_test[grouped_indices[0]], axis=0)
    test_grid_loss_activated_delta = np.mean(delta_test[grouped_indices[1]], axis=0)
    test_grid_temp_activated_delta = np.mean(delta_test[grouped_indices[2]], axis=0)
    test_total_evaluation = np.mean(delta_test, axis=0)

    df = pd.DataFrame(
        {
            "Activated OHE": [
                "Grid Load",
                "Grid Loss",
                "Grid Temp",
                "In total, ignoring OHE",
            ]
            * 3,
            "metric": [
                train_grid_load_activated_delta,
                train_grid_loss_activated_delta,
                train_grid_temp_activated_delta,
                train_total_evaluation,
            ]
            + [
                validation_grid_load_activated_delta,
                validation_grid_loss_activated_delta,
                validation_grid_temp_activated_delta,
                validation_total_evaluation,
            ]
            + [
                test_grid_load_activated_delta,
                test_grid_loss_activated_delta,
                test_grid_temp_activated_delta,
                test_total_evaluation,
            ],
            "Dataset": ["Train"] * 4 + ["Validation"] * 4 + ["Test"] * 4,
        }
    )

    plot = plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Activated OHE", y="metric", hue="Dataset", data=df)

    plt.xticks(rotation=45, ha="right")

    for container in ax.containers:
        if isinstance(container, BarContainer):
            labels = ax.bar_label(container, fmt="%.4f", label_type="edge", padding=3)
            for label in labels:
                label.set_rotation(90)

    plt.title(f"Forecasting SPACE - {metric_name} for each feature based on OHE values")
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Feature")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)

    plt.tight_layout()

    plot.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"retrain_on_{retrain_on}_OHE_Forecasting_{metric_name}.png",
        ),
        dpi=600,
    )
    plt.close(plot)
