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
    sns.histplot(data=data, x="mse", hue="Type", kde=True, alpha=0.7)

    ax.set_title("Feature-wise MSE Distribution")
    ax.set_xlabel("Mean Squared Error (MSE)")
    ax.set_ylabel("Density")

    return fig
