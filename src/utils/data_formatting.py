import pandas as pd


def use_model_predictions_to_create_dataframe(
    predictions, TARGET_NAMES, target_dataframe
):
    column_names = [col.replace("target_", "") for col in TARGET_NAMES]
    predictions_df = pd.DataFrame(predictions, columns=column_names)
    predictions_df["prediction_index"] = target_dataframe.index
    return predictions_df
