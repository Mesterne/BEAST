from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_distribution_of_feature_wise_error(differences_df):
    """
    A plot to show the distribution of error over each distinct time series feature
    in an MTS.
    """

    num_features = len(differences_df.columns)
    fig = make_subplots(
        rows=1,
        cols=num_features,
        subplot_titles=differences_df.columns,
    )

    for i, col in enumerate(differences_df.columns, start=1):
        fig.add_trace(
            go.Histogram(x=differences_df[col], name=col, marker=dict(color="blue")),
            row=1,
            col=i,
        )

    fig.update_layout(
        height=400,
        width=300 * num_features,
        title_text="Distribution of Differences for Prediction Features",
        showlegend=False,
    )

    fig.show()
