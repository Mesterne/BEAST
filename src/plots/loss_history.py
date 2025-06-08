import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Code partially from ChatGPT
def plot_loss_history(train_loss_history, validation_loss_history):
    epochs = len(train_loss_history)

    epochs_range = list(range(1, epochs + 1))

    data = pd.DataFrame(
        {
            "Epoch": epochs_range * 2,
            "Loss": train_loss_history + validation_loss_history,
            "Type": ["Training Loss"] * epochs + ["Validation Loss"] * epochs,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x="Epoch", y="Loss", hue="Type", marker="o", ax=ax)
    ax.set_title("Training and Validation Loss History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()

    return fig
