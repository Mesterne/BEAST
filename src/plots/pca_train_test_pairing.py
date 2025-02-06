import plotly.express as px
import plotly.graph_objects as go


def pca_plot_train_test_pairing(mts_pca_df, dataset_row):
    """
    Plots PCA components for train-test pairings in the dataset.

    This function visualizes two highlighted points in the PCA plot:
    the original index and the target index, connected by a dotted green arrow.

    Parameters:
    mts_pca_df (pd.DataFrame): A DataFrame containing PCA components and the 'index' column.
    dataset_row (pd.DataFrame): A DataFrame with 'original_index' and 'target_index' for pair visualization.

    Returns:
    None: Displays an interactive plot showing the PCA points and highlighted transitions.
    """
    # Convert column names to avoid issues
    dataset_row.columns = [col.replace(" ", "_") for col in dataset_row.columns]

    # Extract the necessary indices
    original_idx = dataset_row.loc[0, "original_index"]
    target_idx = dataset_row.loc[0, "target_index"]

    # Filter the PCA DataFrame for the points to highlight
    original_point = mts_pca_df[mts_pca_df["index"] == original_idx].iloc[0]
    target_point = mts_pca_df[mts_pca_df["index"] == target_idx].iloc[0]

    # Create the base scatter plot
    fig = px.scatter(
        mts_pca_df,
        x="pca1",
        y="pca2",
        hover_data=["index"],
        color="isTrain",
        color_discrete_map={True: "blue", False: "grey"},
    )

    # Add yellow dot for the original point
    fig.add_trace(
        go.Scatter(
            x=[original_point["pca1"]],
            y=[original_point["pca2"]],
            mode="markers",
            marker=dict(color="green", size=15),
            name="Original Index",
        )
    )

    # Add red dot for the target point
    fig.add_trace(
        go.Scatter(
            x=[target_point["pca1"]],
            y=[target_point["pca2"]],
            mode="markers",
            marker=dict(color="red", size=15),
            name="Target Index",
        )
    )

    # Add a dotted arrow between points
    fig.add_trace(
        go.Scatter(
            x=[original_point["pca1"], target_point["pca1"]],
            y=[original_point["pca2"], target_point["pca2"]],
            mode="lines",
            line=dict(color="green", dash="dot"),
            name="Target Transition",
        )
    )

    fig.show()


def pca_plot_train_test_pairing_with_predictions(
    mts_pca_df, dataset_row, predictions, prediction_sample
):
    # Extract the necessary indices
    original_idx = dataset_row.loc["original_index"]
    target_idx = dataset_row.loc["target_index"]

    # Filter the PCA DataFrame for the points to highlight
    original_point = mts_pca_df[mts_pca_df["index"] == original_idx].iloc[0]
    target_point = mts_pca_df[mts_pca_df["index"] == target_idx].iloc[0]

    # Create the base scatter plot
    fig = px.scatter(
        mts_pca_df,
        x="pca1",
        y="pca2",
        hover_data=["index"],
        color="isTrain",
        color_discrete_map={True: "blue", False: "grey"},
    )

    # Add yellow dot for the original point
    fig.add_trace(
        go.Scatter(
            x=[original_point["pca1"]],
            y=[original_point["pca2"]],
            mode="markers",
            marker=dict(color="yellow", size=15),
            name="Original Index",
        )
    )

    # Add red dot for the target point
    fig.add_trace(
        go.Scatter(
            x=[target_point["pca1"]],
            y=[target_point["pca2"]],
            mode="markers",
            marker=dict(color="red", size=15),
            name="Target Index",
        )
    )

    # Add a dotted arrow between original and target points
    fig.add_trace(
        go.Scatter(
            x=[original_point["pca1"], target_point["pca1"]],
            y=[original_point["pca2"], target_point["pca2"]],
            mode="lines",
            line=dict(color="red", dash="dot"),
            name="Target Transition",
        )
    )

    # Step 3: Get the predicted point from the PCA DataFrame
    prediction_point = prediction_sample

    fig.add_trace(
        go.Scatter(
            x=predictions["pca1"],
            y=predictions["pca2"],
            mode="markers",
            marker=dict(color="orange", size=10),
            name="Other predicted points",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[prediction_point["pca1"][0]],
            y=[prediction_point["pca2"][0]],
            mode="markers",
            marker=dict(color="teal", size=15),
            name="Prediction",
        )
    )

    # Step 5: Add a line from the original point to the prediction point (green)
    fig.add_trace(
        go.Scatter(
            x=[original_point["pca1"], prediction_point["pca1"][0]],
            y=[original_point["pca2"], prediction_point["pca2"][0]],
            mode="lines",
            line=dict(color="green", dash="dot"),
            name="Original to Prediction",
        )
    )

    fig.show()
