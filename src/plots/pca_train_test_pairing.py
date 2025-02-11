import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def pca_plot_train_test_pairing(mts_pca_df: pd.DataFrame, dataset_row: pd.DataFrame):
    """
    Plots PCA components for train-test pairings in the dataset.

    This function visualizes two highlighted points in the PCA plot:
    the original index and the target index, connected by a dotted green arrow.

    Parameters:
    mts_pca_df (pd.DataFrame): A DataFrame containing PCA components and the 'index' column.
    dataset_row (pd.DataFrame): A DataFrame with 'original_index' and 'target_index' for pair visualization.

    Returns:
    fig (matplotlib.figure.Figure): Matplotlib figure showing the PCA points and highlighted transitions.
    """
    # Ensure column names are compatible
    dataset_row.columns = [col.replace(" ", "_") for col in dataset_row.columns]

    # Extract relevant indices
    original_idx = dataset_row.loc[0, "original_index"]
    target_idx = dataset_row.loc[0, "target_index"]

    # Filter points for highlighting
    original_point = mts_pca_df[mts_pca_df["index"] == original_idx].iloc[0]
    target_point = mts_pca_df[mts_pca_df["index"] == target_idx].iloc[0]

    # Assign labels and colors based on conditions
    mts_pca_df["category"] = mts_pca_df.apply(
        lambda row: (
            "Train"
            if row["isTrain"]
            else (
                "Validation"
                if row["isValidation"]
                else "Test" if row["isTest"] else "Other"
            )
        ),
        axis=1,
    )

    # Plot scatter points using seaborn
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.scatterplot(
        data=mts_pca_df,
        x="pca1",
        y="pca2",
        hue="category",
        palette={
            "Train": "blue",
            "Validation": "grey",
            "Test": "red",
            "Other": "black",
        },
        legend="full",
        s=50,
        ax=ax,
    )

    # Highlight original point
    ax.scatter(
        original_point["pca1"],
        original_point["pca2"],
        color="orange",
        s=150,
        label="Original Index",
        edgecolor="black",
    )

    # Highlight target point
    ax.scatter(
        target_point["pca1"],
        target_point["pca2"],
        color="purple",
        s=150,
        label="Target Index",
        edgecolor="black",
    )

    # Add dotted arrow between points
    ax.plot(
        [original_point["pca1"], target_point["pca1"]],
        [original_point["pca2"], target_point["pca2"]],
        linestyle="dotted",
        color="orange",
        label="Target Transition",
    )

    ax.set_title("PCA Plot with Train-Test Pairing")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()

    return fig


def pca_plot_train_test_pairing_with_predictions(
    mts_pca_df, dataset_row, predictions, prediction_sample
):
    # Extract relevant indices
    original_idx = dataset_row.loc["original_index"]
    target_idx = dataset_row.loc["target_index"]

    # Filter the PCA DataFrame for the points to highlight
    original_point = mts_pca_df[mts_pca_df["index"] == original_idx].iloc[0]
    target_point = mts_pca_df[mts_pca_df["index"] == target_idx].iloc[0]

    # Assign labels and colors based on conditions
    mts_pca_df["category"] = mts_pca_df.apply(
        lambda row: (
            "Train"
            if row["isTrain"]
            else (
                "Validation"
                if row["isValidation"]
                else "Test" if row["isTest"] else "Other"
            )
        ),
        axis=1,
    )

    # Plot scatter points using seaborn
    fig, ax = plt.subplots(figsize=(12, 8))

    # Base scatter plot for categories
    sns.scatterplot(
        data=mts_pca_df,
        x="pca1",
        y="pca2",
        hue="category",
        palette={
            "Train": "blue",
            "Validation": "grey",
            "Test": "red",
            "Other": "black",
        },
        legend="full",
        s=50,
        ax=ax,
    )

    # Plot key points
    ax.scatter(
        original_point["pca1"],
        original_point["pca2"],
        color="yellow",
        s=150,
        label="Original Index",
        edgecolor="black",
    )

    ax.scatter(
        target_point["pca1"],
        target_point["pca2"],
        color="red",
        s=150,
        label="Target Index",
        edgecolor="black",
    )

    # Add dotted arrow between original and target points
    ax.plot(
        [original_point["pca1"], target_point["pca1"]],
        [original_point["pca2"], target_point["pca2"]],
        linestyle="dotted",
        color="red",
        label="Target Transition",
    )

    # Get the predicted point from the PCA DataFrame
    prediction_point = prediction_sample

    ax.scatter(
        predictions["pca1"],
        predictions["pca2"],
        color="orange",
        s=100,
        label="Other Predicted Points",
        edgecolor="black",
    )

    ax.scatter(
        prediction_point["pca1"][0],
        prediction_point["pca2"][0],
        color="teal",
        s=150,
        label="Prediction",
        edgecolor="black",
    )

    # Add line from original point to the prediction point (green)
    ax.plot(
        [original_point["pca1"], prediction_point["pca1"][0]],
        [original_point["pca2"], prediction_point["pca2"][0]],
        linestyle="dotted",
        color="green",
        label="Original to Prediction",
    )

    ax.set_title("PCA Plot with Train/Test Pairing and Predictions")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()

    return fig
