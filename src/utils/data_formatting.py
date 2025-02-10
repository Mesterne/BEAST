from turtle import pd
from src.training.feedforward_features import TARGET_NAMES


def use_model_predictions_to_create_dataframe(predictions):
    column_names = [col.replace("target_", "") for col in TARGET_NAMES]
    predictions_df = pd.DataFrame(predictions, columns=column_names)
    predictions_df["prediction_index"] = range(len(predictions))
    return predictions_df
