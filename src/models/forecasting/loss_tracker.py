from pytorch_lightning.callbacks import Callback


class LossTracker(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if "train_loss" in logs:
            self.train_loss.append(logs["train_loss"].cpu().detach().item())
        if "val_loss" in logs:
            self.val_loss.append(logs["val_loss"].cpu().detach().item())
