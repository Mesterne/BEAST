import numpy as np


class ModelHandler:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = None

    def choose_model_category(self, mts_dataset: np.ndarray):
        X_train, y_train, X_val, y_val = self.model.create_training_data(mts_dataset)
        self.model.train(X_train, y_train, X_val, y_val)
        pass

    def train():
        assert self.model is not None, "The model to train must be defined"

    def infer():
        """ """
        assert self.model is not None, "The model must be defined"
