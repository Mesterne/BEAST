import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_loss_history(train_loss_history, validation_loss_history, epochs):
    # Prepare data for plotting
    epochs_range = list(range(1, epochs + 1))

    # Combine data for both training and validation loss
    data = pd.DataFrame(
        {
            "Epoch": epochs_range * 2,
            "Loss": train_loss_history + validation_loss_history,
            "Type": ["Training Loss"] * epochs + ["Validation Loss"] * epochs,
        }
    )

    # Plot combined Training and Validation Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x="Epoch", y="Loss", hue="Type", marker="o", ax=ax)
    ax.set_title("Training and Validation Loss History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()

    return fig
