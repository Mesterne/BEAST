import pandas as pd
import numpy as np
from tqdm import tqdm


class CorrelationModel:
    def train(self, training_set):
        """
        To train the model, we calculate the correlations between the time series features
        in the training set.
        """
        self.correlations = training_set.corr()

    def infer(self, values: pd.DataFrame):
        """
        Inference of the model is based on a naive correlation assumption.
        We change all other features except the one that is transformed.
        To do this, we record the delta of the original transformation. We then
        update each feature by adding this delta multiplied with the correlation factor between the features.
        """
        # We copy the inputted values to ensure the method does not change the structure inplace
        updated_values = values.copy()
        predictions = pd.DataFrame()

        # We only keep the columns that are prefixed by 'original_' or 'delta_'
        columns_to_keep = [
            col
            for col in updated_values.columns
            if col.startswith("original_") or col.startswith("delta_")
        ]
        updated_values = updated_values[columns_to_keep]

        for idx, row in tqdm(updated_values.iterrows(), total=len(updated_values)):
            # Find the delta column with the largest absolute value
            delta_columns = [col for col in row.index if col.startswith("delta_")]
            if not delta_columns:
                continue

            # Find the column with the largest absolute delta
            max_delta_col = max(delta_columns, key=lambda col: abs(row[col]))
            max_delta = row[max_delta_col]
            feature_name = max_delta_col[len("delta_") :]  # Extract feature name

            # If a valid delta is found, make a prediction
            if max_delta != 0 and feature_name:
                row_with_originals = values.loc[
                    idx,
                    [
                        col
                        for col in values.columns
                        if col.startswith("original_") and col != "original_index"
                    ],
                ]
                row_with_originals.index = [
                    col[len("original_") :] for col in row_with_originals.index
                ]

                prediction = (
                    row_with_originals + max_delta * self.correlations[feature_name]
                )

                for col in prediction.index:
                    if "trend-slope" in col:
                        prediction[col] = prediction[col].clip(-1, 1)
                    else:
                        prediction[col] = prediction[col].clip(0, 1)

                prediction["prediction_index"] = int(idx)
                predictions = pd.concat(
                    [predictions, prediction.to_frame().T], ignore_index=True
                )

        return predictions
