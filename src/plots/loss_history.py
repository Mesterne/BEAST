import plotly.graph_objects as go


def plot_loss_history(train_loss_history, validation_loss_history, epochs):
    fig = go.Figure()

    # Plot Training Loss
    fig.add_trace(
        go.Scatter(
            x=list(range(1, epochs + 1)),
            y=train_loss_history,
            mode="lines",
            name="Training Loss",
        )
    )

    # Plot Validation Loss
    fig.add_trace(
        go.Scatter(
            x=list(range(1, epochs + 1)),
            y=validation_loss_history,
            mode="lines",
            name="Validation Loss",
        )
    )

    fig.update_layout(
        title="Training and Validation Loss History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
    )
    return fig
