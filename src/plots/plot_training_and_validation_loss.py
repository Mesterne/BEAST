import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.constants import OUTPUT_DIR


def plot_training_and_validation_loss(
    training_loss: np.ndarray,
    validation_loss: np.ndarray,
    model_name: str = "unnamed_model",
) -> None:
    epochs = np.arange(1, len(training_loss) + 1)
    data = pd.DataFrame(
        {
            "Epoch": np.concatenate([epochs, epochs]),
            "Loss": np.concatenate([training_loss, validation_loss]),
            "Type": ["Training"] * len(training_loss)
            + ["Validation"] * len(validation_loss),
        }
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="Epoch", y="Loss", hue="Type", marker="o")

    plt.title(f"Training and Validation Loss - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(title="Loss Type")
    plt.tight_layout()

    # Save the plot as PNG
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_loss_plot.png"))
    plt.close()
