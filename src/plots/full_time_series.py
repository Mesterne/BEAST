import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.constants import COLUMN_NAMES, UTS_NAMES
from src.utils.logging_config import logger


def plot_time_series_for_all_uts(
    original_mts: np.array,  # Shape: (num_uts, num_time_steps)
    target_mts: np.array,
    transformed_mts: np.array,
    original_mts_features: np.array,  # Shape: (num_uts*num_features)
    target_mts_features: np.array,
    predicted_mts_features: np.array,
    transformed_mts_features: np.array,
) -> plt.Figure:
    """
    Plots each UTS as a row where columns represent original, target, and transformed MTS.
    Also plots feature distributions.

    Parameters:
    - original_mts, target_mts, transformed_mts: NumPy arrays (num_uts, num_time_steps)
    - original_mts_features, target_mts_features, transformed_mts_features: NumPy arrays (num_uts, num_features)
    - uts_names: List of UTS names (same length as num_uts)
    - feature_names: List of feature names (same length as num_features)

    Returns:
    - fig (matplotlib.figure.Figure): The generated plot.
    """

    num_uts = original_mts.shape[0]
    num_features = len(COLUMN_NAMES) // num_uts
    num_time_steps = original_mts.shape[1]

    fig, axes = plt.subplots(nrows=num_uts, ncols=4, figsize=(30, 5 * num_uts))

    transformed_mts_features = transformed_mts_features.squeeze()
    time_index = np.arange(num_time_steps)  # Time steps for x-axis

    original_mts_features = original_mts_features.reshape((num_uts, num_features))
    target_mts_features = target_mts_features.reshape((num_uts, num_features))
    transformed_mts_features = transformed_mts_features.reshape((num_uts, num_features))
    predicted_mts_features = predicted_mts_features.reshape((num_uts, num_features))

    for i, uts_name in enumerate(UTS_NAMES):
        # Plot time series
        sns.lineplot(x=time_index, y=original_mts[i, :], ax=axes[i, 0], color="red")
        sns.lineplot(x=time_index, y=target_mts[i, :], ax=axes[i, 1], color="blue")
        sns.lineplot(
            x=time_index, y=transformed_mts[i, :], ax=axes[i, 2], color="purple"
        )

        axes[i, 0].set_title(f"{uts_name} - Original")
        axes[i, 1].set_title(f"{uts_name} - Target")
        axes[i, 2].set_title(f"{uts_name} - Transformed")

        # Hide x-axis labels
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].xaxis.set_visible(False)

        # Extract relevant features for the current UTS
        grouped_features = {}

        # Loop over the available number of features (i.e., num_features)
        for j, feature_name in enumerate(
            COLUMN_NAMES
        ):  # j should not exceed the number of features

            if uts_name in feature_name:
                base_name = "_".join(
                    feature_name.split("_")[1:]
                )  # Extract base feature name

                if base_name not in grouped_features:
                    grouped_features[base_name] = {
                        "original": [],
                        "target": [],
                        "transformed": [],
                        "predicted": [],
                    }

                grouped_features[base_name]["original"] = original_mts_features[i][
                    j % num_features
                ]
                grouped_features[base_name]["target"] = target_mts_features[i][
                    j % num_features
                ]
                grouped_features[base_name]["transformed"] = transformed_mts_features[
                    i
                ][j % num_features]
                grouped_features[base_name]["predicted"] = predicted_mts_features[i][
                    j % num_features
                ]

        # Prepare DataFrame for seaborn bar plot
        data = []
        for base_name, values in grouped_features.items():
            data.append(
                {"Feature": base_name, "Type": "original", "Value": values["original"]}
            )
            data.append(
                {"Feature": base_name, "Type": "target", "Value": values["target"]}
            )
            data.append(
                {
                    "Feature": base_name,
                    "Type": "transformed",
                    "Value": values["transformed"],
                }
            )
            data.append(
                {
                    "Feature": base_name,
                    "Type": "predicted",
                    "Value": values["predicted"],
                }
            )

        # Create DataFrame only if there is data to plot
        if data:
            df = pd.DataFrame(data)

            # Plot feature values
            sns.barplot(x="Feature", y="Value", hue="Type", data=df, ax=axes[i, 3])
            axes[i, 3].set_title(f"{uts_name} - Features")
            axes[i, 3].set_xticks(axes[i, 3].get_xticks())
            axes[i, 3].set_xticklabels(axes[i, 3].get_xticklabels(), rotation=45)
        else:
            logger.warning(f"No features found for {uts_name}, skipping feature plot.")

    plt.tight_layout()
    return fig
