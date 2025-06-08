import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.constants import OUTPUT_DIR
from src.utils.logging_config import logger


# Code partially from ChatGPT
def plot_training_and_validation_loss(
    training_loss: np.ndarray, validation_loss: np.ndarray, model_name: str
) -> None:

    training_loss = np.log(training_loss)
    validation_loss = np.log(validation_loss)
    epochs = np.arange(1, len(training_loss) + 1)
    data = pd.DataFrame(
        {
            "Epoch": np.concatenate([epochs, epochs]),
            "Loss": np.concatenate([training_loss, validation_loss]),
            "Type": ["Training"] * len(training_loss)
            + ["Validation"] * len(validation_loss),
        }
    )
    logger.info(
        f"Plotting train/validation loss for model: {model_name}, for {len(epochs)} epochs"
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="Epoch", y="Loss", hue="Type")

    plt.title(f"Training and Validation Loss - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend(title="Loss Type")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_loss_plot.png"), dpi=600)
    plt.close()


def plot_detailed_training_loss(
    training_loss: np.ndarray,
    training_loss_kl_divergence: np.ndarray,
    train_loss_reconstruction: np.ndarray,
    model_name: str = "unnamed_model",
) -> None:
    training_loss = np.log(training_loss)
    training_loss_kl_divergence = np.log(training_loss_kl_divergence)
    train_loss_reconstruction = np.log(train_loss_reconstruction)
    epochs = np.arange(1, len(training_loss) + 1)
    data = pd.DataFrame(
        {
            "Epoch": np.concatenate([epochs, epochs, epochs]),
            "Loss": np.concatenate(
                [training_loss, training_loss_kl_divergence, train_loss_reconstruction]
            ),
            "Type": ["Training"] * len(training_loss)
            + ["Training KL divergence"] * len(training_loss_kl_divergence)
            + ["Training Reconstruction"] * len(train_loss_reconstruction),
        }
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="Epoch", y="Loss", hue="Type")

    plt.title(f"Detailed training loss - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(title="Loss Type")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{model_name}_detailed_training_loss_plot.png"),
        dpi=600,
    )
    plt.close()
