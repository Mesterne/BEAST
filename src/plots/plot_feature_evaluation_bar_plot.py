import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer

from src.data.constants import PLOT_NAMES


def plot_metric_for_each_feature_bar_plot(
    intermediate_evaluation_for_each_feature: np.ndarray,
    final_evaluation_for_each_feature: np.ndarray,
    dataset: str,
    metric: str,
):
    mean_intermediate = np.mean(intermediate_evaluation_for_each_feature, axis=0)
    mean_final = np.mean(final_evaluation_for_each_feature, axis=0)

    df = pd.DataFrame(
        {
            "Inferred or generated": ["Inferred"] * len(mean_intermediate)
            + ["After MTS Generation"] * len(mean_final),
            "Feature": PLOT_NAMES + PLOT_NAMES,
            "metric": np.concatenate([mean_intermediate, mean_final]),
        }
    )
    plot = plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Feature", y="metric", hue="Inferred or generated", data=df)

    plt.xticks(rotation=45, ha="right")

    for container in ax.containers:
        if isinstance(container, BarContainer):
            labels = ax.bar_label(container, fmt="%.4f", label_type="edge", padding=3)
            for label in labels:
                label.set_rotation(90)

    plt.title(f"{dataset} {metric} for each feature. Inferred and Final")
    plt.ylabel(f"{metric}")
    plt.xlabel("Feature")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)

    plt.tight_layout()

    return plot
