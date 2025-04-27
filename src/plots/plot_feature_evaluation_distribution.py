import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_feature_mse_distribution(
    feature_space_mse_validation: np.ndarray, feature_space_mse_test: np.ndarray
) -> Figure:
    """
    Creates distribution plot of the MSE values for each MTS feature predictions.
    args:
        feature_space_mse_validation: np.ndarray - 1 dimensional numpy array of the MSE values of the feature model.
        feature_space_mse_test: np.ndarray - 1 dimensional numpy array of the MSE values of the feature model.
    """
    data = pd.DataFrame(
        {
            "mse": np.concatenate(
                [feature_space_mse_validation, feature_space_mse_test]
            ),
            "Type": ["Validation"] * len(feature_space_mse_validation)
            + ["Test"] * len(feature_space_mse_test),
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=data, x="mse", hue="Type", fill=True, ax=ax)

    ax.set_title("Feature-wise MSE Distribution")
    ax.set_xlabel("Mean Squared Error (MSE)")
    ax.set_ylabel("Density")

    return fig


def plot_feature_evaluation(
    inferred_feature_space_mse_validation: float,
    final_feature_space_mse_validation: float,
    inferred_feature_space_mse_test: float,
    final_feature_space_mse_test: float,
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
                inferred_feature_space_mse_validation,
                final_feature_space_mse_validation,
                inferred_feature_space_mse_test,
                final_feature_space_mse_test,
            ],
        }
    )
    sns.barplot(x="Dataset", y="metric", hue="Type", data=df)
    plt.title(f"{metric_name} Comparison: Old vs New Model")
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Dataset")

    return plot
