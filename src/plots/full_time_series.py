import matplotlib.pyplot as plt
import seaborn as sns


def plot_time_series_for_all_uts(
    original_mts,
    target_mts,
    transformed_mts,
):
    # Original, target and transformed MTS have columns which corresponds to one uts. The index is time.
    # Use seaborn to plot each uts as a row in the figure where the columns are original, target and transformed MTS.

    num_uts = len(original_mts.columns)
    fig, axes = plt.subplots(nrows=num_uts, ncols=3, figsize=(15, 5 * num_uts))
    uts_names = original_mts.columns

    for i, uts_name in enumerate(uts_names):
        sns.lineplot(x=original_mts.index, y=original_mts[uts_name], ax=axes[i, 0])
        sns.lineplot(x=target_mts.index, y=target_mts[uts_name], ax=axes[i, 1])
        sns.lineplot(
            x=transformed_mts.index, y=transformed_mts[uts_name], ax=axes[i, 2]
        )

        axes[i, 0].set_title(f"{uts_name} - Original")
        axes[i, 1].set_title(f"{uts_name} - Target")
        axes[i, 2].set_title(f"{uts_name} - Transformed")

    plt.tight_layout()
    return fig
