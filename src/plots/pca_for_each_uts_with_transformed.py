import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils.pca import PCAWrapper


def plot_pca_for_each_uts_with_transformed(
    mts_features_df,
    transformed_features,
    predicted_features,
    original_index,
    target_index,
    uts_names,
):
    num_uts = len(uts_names)
    fig, axes = plt.subplots(
        nrows=num_uts + 1, ncols=1, figsize=(12, 8 * (num_uts + 1))
    )

    if num_uts == 1:
        axes = [axes]

    # Plot for the entire MTS first
    ax = axes[0]
    pca_transformer = PCAWrapper(n_components=2)

    tmp = pca_transformer.fit_transform(mts_features_df)
    pca_df = pd.DataFrame(tmp, columns=["pca1", "pca2"])
    pca_df["index"] = mts_features_df.index

    transformed_pca = pca_transformer.transform(transformed_features)[["pca1", "pca2"]]
    predicted_pca = pca_transformer.transform(predicted_features)[["pca1", "pca2"]]

    original_point = mts_features_df.iloc[[original_index]]
    target_point = mts_features_df.iloc[[target_index]]
    original_point.loc[:, ["pca1", "pca2"]] = pca_transformer.transform(original_point)
    target_point.loc[:, ["pca1", "pca2"]] = pca_transformer.transform(target_point)

    # Plot scatter points using seaborn
    sns.scatterplot(
        data=pca_df,
        x="pca1",
        y="pca2",
        color="grey",
        s=50,
        ax=ax,
    )

    # Highlight original point
    ax.scatter(
        original_point["pca1"],
        original_point["pca2"],
        color="blue",
        s=150,
        label="Original Index",
        edgecolor="black",
    )

    # Highlight target point
    ax.scatter(
        target_point["pca1"],
        target_point["pca2"],
        color="green",
        s=150,
        label="Target Index",
        edgecolor="black",
    )
    ax.scatter(
        transformed_pca["pca1"],
        transformed_pca["pca2"],
        color="red",
        s=150,
        label="GA transformed Index",
        edgecolor="black",
    )
    ax.scatter(
        predicted_pca["pca1"],
        predicted_pca["pca2"],
        color="orange",
        s=150,
        label="Predicted Index",
        edgecolor="black",
    )

    ax.set_title("PCA Plot with transformed MTS")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()

    # Plot for each UTS
    for ax, uts in zip(axes[1:], uts_names):
        # Filter columns for the current UTS
        uts_columns = [col for col in mts_features_df.columns if uts in col]
        uts_features_df = mts_features_df[uts_columns]
        uts_transformed_features = transformed_features[uts_columns]
        uts_predicted_features = predicted_features[uts_columns]

        # Filter points for highlighting
        original_point = uts_features_df.iloc[[original_index]]
        target_point = uts_features_df.iloc[[target_index]]

        pca_transformer = PCAWrapper(n_components=2)

        tmp = pca_transformer.fit_transform(uts_features_df)
        pca_df = pd.DataFrame(tmp, columns=["pca1", "pca2"])
        pca_df["index"] = uts_features_df.index

        transformed_pca = pca_transformer.transform(uts_transformed_features)[
            ["pca1", "pca2"]
        ]
        predicted_pca = pca_transformer.transform(uts_predicted_features)[
            ["pca1", "pca2"]
        ]

        original_point.loc[:, ["pca1", "pca2"]] = pca_transformer.transform(
            original_point
        )
        target_point.loc[:, ["pca1", "pca2"]] = pca_transformer.transform(target_point)

        # Plot scatter points using seaborn
        sns.scatterplot(
            data=pca_df,
            x="pca1",
            y="pca2",
            color="grey",
            s=50,
            ax=ax,
        )

        # Highlight original point
        ax.scatter(
            original_point["pca1"],
            original_point["pca2"],
            color="blue",
            s=150,
            label="Original Index",
            edgecolor="black",
        )

        # Highlight target point
        ax.scatter(
            target_point["pca1"],
            target_point["pca2"],
            color="green",
            s=150,
            label="Target Index",
            edgecolor="black",
        )
        ax.scatter(
            transformed_pca["pca1"],
            transformed_pca["pca2"],
            color="red",
            s=150,
            label="GA transformed Index",
            edgecolor="black",
        )
        ax.scatter(
            predicted_pca["pca1"],
            predicted_pca["pca2"],
            color="orange",
            s=150,
            label="Predicted Index",
            edgecolor="black",
        )

        ax.set_title(f"PCA Plot with transformed UTS for {uts}")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.legend()

    plt.tight_layout()
    return fig
