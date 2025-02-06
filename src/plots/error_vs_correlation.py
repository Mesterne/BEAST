import plotly.graph_objects as go


def plot_error_vs_correlation_for_each_feature(
    mse_values_for_each_feature, correlation_matrix
):
    """
    Plots error values and correlations for different feature groups.

    This function creates two visualizations:
    1. A bar plot showing the Mean Squared Error (MSE) for each feature.
    2. A bar plot showing the correlations of each feature with PCA components (PCA1 and PCA2).

    Parameters:
    mse_values_for_each_feature (dict): A dictionary mapping feature names to their MSE values.
    correlation_matrix (pd.DataFrame): A DataFrame containing correlation values, indexed by feature names,
        including correlations with 'pca1' and 'pca2'.

    Returns:
    None: Displays interactive plots in the output.
    """
    pca_corr = correlation_matrix.drop(["pca1", "pca2"])

    groups = {
        "seasonal_strength": [
            "grid-load_seasonal-strength",
            "grid-loss_seasonal-strength",
            "grid-temp_seasonal-strength",
        ],
        "trend_linearity": [
            "grid-load_trend-linearity",
            "grid-loss_trend-linearity",
            "grid-temp_trend-linearity",
        ],
        "trend_slope": [
            "grid-load_trend-slope",
            "grid-loss_trend-slope",
            "grid-temp_trend-slope",
        ],
        "trend_strength": [
            "grid-load_trend-strength",
            "grid-loss_trend-strength",
            "grid-temp_trend-strength",
        ],
    }

    x_vals = []
    y_vals_mse = []
    y_vals_pca1 = []
    y_vals_pca2 = []
    bar_labels = []

    offset = 0
    group_width = 0.2

    for group, columns in groups.items():
        for i, column in enumerate(columns):
            x_vals.append(offset + i * group_width)  # Position the bars in each group
            y_vals_mse.append(
                mse_values_for_each_feature[column]
            )  # MSE values for each column
            y_vals_pca1.append(pca_corr.loc[column, "pca1"])  # Correlation to pca1
            y_vals_pca2.append(pca_corr.loc[column, "pca2"])  # Correlation to pca2
            bar_labels.append(column)  # Store the column name for labels
        offset += len(columns) * group_width + group_width  # Move to the next group

    x_vals.append(offset)
    bar_labels.append("Overall")

    fig_mse = go.Figure()

    fig_mse.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals_mse,
            name="MSE by Column",
            text=bar_labels,
            hoverinfo="text+y",
        )
    )

    fig_mse.update_layout(
        title="MSE for Each Column",
        xaxis_title="Columns (Grouped)",
        yaxis_title="MSE Value",
        height=600,
        width=1000,
        xaxis=dict(
            tickvals=x_vals,
            ticktext=bar_labels,
            tickangle=45,  # Rotate labels for readability
        ),
        showlegend=True,
    )

    fig_mse.show()

    fig_corr = go.Figure()

    fig_corr.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals_pca1,
            name="Correlation to PCA1",
            text=bar_labels,
            hoverinfo="text+y",
            marker_color="blue",
        )
    )

    fig_corr.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals_pca2,
            name="Correlation to PCA2",
            text=bar_labels,
            hoverinfo="text+y",
            marker_color="red",
        )
    )

    fig_corr.update_layout(
        title="Correlation to PCA1 and PCA2 for Each Column",
        xaxis_title="Columns (Grouped)",
        yaxis_title="Correlation Value",
        height=600,
        width=1000,
        xaxis=dict(tickvals=x_vals, ticktext=bar_labels, tickangle=45),
        showlegend=True,
    )
    fig_corr.show()
