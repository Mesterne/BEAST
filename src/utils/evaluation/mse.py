def get_mse_for_features_and_overall(differences_df):
    mse_values_for_each_feature = (
        differences_df.drop(columns=["prediction_index"]) ** 2
    ).mean()
    overall_mse = mse_values_for_each_feature.mean()

    return overall_mse, mse_values_for_each_feature
