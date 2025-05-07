import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer
from matplotlib.figure import Figure


def plot_feature_evaluation(
    intermediate_mse_values_for_each_feature_validation: np.ndarray,
    intermediate_mse_values_for_each_feature_test: np.ndarray,
    final_mse_values_for_each_feature_validation: np.ndarray,
    final_mse_values_for_each_feature_test: np.ndarray,
    metric_name: str = "",
) -> Figure:
    """
    Creates bar plot with evaluation over inferred features for validation and test set.
    args:
       feature_space_mse_validation: float - MSE over inferred features for validation
       feature_space_mse_test: float - MSE over inferred features for test
    """
    plot = plt.figure(figsize=(10, 6))

    df = pd.DataFrame(
        {
            "Type": [
                "Inferred",
                "Final MTS Features",
                "Inferred",
                "Final MTS Features",
            ],
            "Dataset": ["Validation", "Validation", "Test", "Test"],
            "metric": [
                np.mean(intermediate_mse_values_for_each_feature_validation),
                np.mean(final_mse_values_for_each_feature_validation),
                np.mean(intermediate_mse_values_for_each_feature_test),
                np.mean(final_mse_values_for_each_feature_test),
            ],
        }
    )
    ax = sns.barplot(x="Dataset", y="metric", hue="Type", data=df)

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.4f", label_type="edge", padding=3)

    plt.title(f"{metric_name} Comparison: Old vs New Model")
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Dataset")

    return plot
