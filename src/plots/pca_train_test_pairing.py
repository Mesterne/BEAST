import plotly.express as px
import plotly.graph_objects as go

def pca_plot_train_test_pairing(mts_pca_df, dataset_row):
    # Convert column names to avoid issues
    dataset_row.columns = [col.replace(" ", "_") for col in dataset_row.columns]

    # Extract the necessary indices
    original_idx = dataset_row.loc[0, 'original_index']
    target_idx = dataset_row.loc[0, 'target_index']

    # Filter the PCA DataFrame for the points to highlight
    original_point = mts_pca_df[mts_pca_df['index'] == original_idx].iloc[0]
    target_point = mts_pca_df[mts_pca_df['index'] == target_idx].iloc[0]

    # Create the base scatter plot
    fig = px.scatter(
        mts_pca_df,
        x='pca1',
        y='pca2',
        hover_data=['index'],
        color='isTrain',
        color_discrete_map={True: 'blue', False: 'grey'}
    )

    # Add yellow dot for the original point
    fig.add_trace(go.Scatter(
        x=[original_point['pca1']],
        y=[original_point['pca2']],
        mode='markers',
        marker=dict(color='green', size=15),
        name='Original Index'
    ))

    # Add red dot for the target point
    fig.add_trace(go.Scatter(
        x=[target_point['pca1']],
        y=[target_point['pca2']],
        mode='markers',
        marker=dict(color='red', size=15),
        name='Target Index'
    ))

    # Add a dotted arrow between points
    fig.add_trace(go.Scatter(
        x=[original_point['pca1'], target_point['pca1']],
        y=[original_point['pca2'], target_point['pca2']],
        mode='lines',
        line=dict(color='green', dash='dot'),
        name='Target Transition'
    ))

    fig.show()