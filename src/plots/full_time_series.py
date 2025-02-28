import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logging_config import logger
import pandas as pd


def plot_time_series_for_all_uts(
    original_mts,
    target_mts,
    transformed_mts,
    original_mts_features,
    target_mts_features,
    transformed_mts_features,
):
    # Original, target and transformed MTS have columns which corresponds to one uts. The index is time.
    # Use seaborn to plot each uts as a row in the figure where the columns are original, target and transformed MTS.

    num_uts = len(original_mts.columns)
    fig, axes = plt.subplots(nrows=num_uts, ncols=4, figsize=(30, 5 * num_uts))
    uts_names = original_mts.columns

    transformed_mts_features = transformed_mts_features.squeeze()

    for i, uts_name in enumerate(uts_names):
        # Plot the time series
        sns.lineplot(
            x=original_mts.index, y=original_mts[uts_name], ax=axes[i, 0], color="red"
        )
        sns.lineplot(
            x=target_mts.index, y=target_mts[uts_name], ax=axes[i, 1], color="green"
        )
        sns.lineplot(
            x=transformed_mts.index,
            y=transformed_mts[uts_name],
            ax=axes[i, 2],
            color="blue",
        )

        axes[i, 0].set_title(f"{uts_name} - Original")
        axes[i, 1].set_title(f"{uts_name} - Target")
        axes[i, 2].set_title(f"{uts_name} - Transformed")

        # Dictionary to store grouped feature values
        grouped_features = {}

        possible_features = original_mts_features.index[
            original_mts_features.index.str.contains(uts_name)
        ]

        for feature_name in possible_features:
            # Extract base feature name (everything after first '_')
            base_name = "_".join(feature_name.split("_")[1:])

            # Initialize if not already present
            if base_name not in grouped_features:
                grouped_features[base_name] = {
                    "original": [],
                    "target": [],
                    "transformed": [],
                }

            # Append feature values
            grouped_features[base_name]["original"].append(
                original_mts_features.loc[feature_name]
            )
            grouped_features[base_name]["target"].append(
                target_mts_features.loc[feature_name]
            )
            grouped_features[base_name]["transformed"].append(
                transformed_mts_features.loc[feature_name]
            )

        # Convert to a format suitable for seaborn
        data = []
        for base_name, values in grouped_features.items():
            data.append(
                {
                    "Feature": base_name,
                    "Type": "original",
                    "Value": sum(values["original"]),
                }
            )
            data.append(
                {"Feature": base_name, "Type": "target", "Value": sum(values["target"])}
            )
            data.append(
                {
                    "Feature": base_name,
                    "Type": "transformed",
                    "Value": sum(values["transformed"]),
                }
            )

        df = pd.DataFrame(data)

        # Plot
        sns.barplot(x="Feature", y="Value", hue="Type", data=df, ax=axes[i, 3])
        plt.xticks(rotation=45)  # Rotate labels for better visibility
        plt.title("Grouped Feature Values")

        axes[i, 3].set_title(f"{uts_name} - Features")

    plt.tight_layout()
    return fig
