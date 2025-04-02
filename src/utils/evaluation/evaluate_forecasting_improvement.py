from typing import Dict


class ForecasterEvaluator:
    def __init__(self, config: Dict[str, any]) -> None:
        self.config = config
        self.mts_dataset = mts_dataset
        self.train_indices = train_indices
        self.validation_indices = validation_indices
        self.test_indices = test_indices
