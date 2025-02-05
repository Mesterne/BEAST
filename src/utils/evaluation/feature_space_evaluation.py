
import pandas as pd
from tqdm import tqdm


def find_error_of_each_feature_for_each_sample(predictions, labelled_test_dataset):
    # Initialize an empty list to store differences
    all_differences = []

    for idx, row in tqdm(predictions.iterrows(), total=len(predictions)):
        prediction_idx = int(row['prediction_index'])
        
        target_row = labelled_test_dataset.loc[prediction_idx]

        prediction_columns = [col for col in predictions.columns if not col.startswith(('pca', 'prediction_index'))]
        target_columns = [col for col in target_row.index if col.startswith('target_')]

        target_row_filtered = target_row[target_columns]
        target_row_filtered.index = target_row_filtered.index.str.replace('^target_', '', regex=True)
        
        differences = {
            col: abs(row[col] - target_row_filtered[col]) 
            for col in prediction_columns if col in target_row_filtered.index
        }
        differences['prediction_index'] = prediction_idx
        all_differences.append(differences)

    differences_df = pd.DataFrame(all_differences)
    return differences_df