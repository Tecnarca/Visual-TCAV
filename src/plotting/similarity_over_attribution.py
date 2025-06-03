import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fontTools.subset import subset

from src.loggers import LOGGER, REPO_ROOT_PATH
from src.plotting.dataframe_from_prettytables_loader import \
    load_prettytables_as_dataframe


def load_dataframe_from_experiment_set(set_prefix: str) -> pd.DataFrame:
    root_dir = REPO_ROOT_PATH / "experiments"

    result = {
        folder.name: [
            f.name for f in folder.iterdir() if f.is_file() and f.suffix == ""
        ]
        for folder in root_dir.iterdir()
        if folder.is_dir() and folder.name.startswith(set_prefix)
    }

    loaded_dataframes = {
        (folder, file): load_prettytables_as_dataframe(root_dir / folder / file)
        for folder, files in result.items()
        for file in files
    }

    concats = pd.concat(
        [
            df.assign(Folder=folder, file=file)
            for (folder, file), df in loaded_dataframes.items()
        ],
        ignore_index=True,
    )
    concats = concats[concats["Concept"] != "random"]
    return concats.drop(columns=["CI Low", "CI High", "file"]).drop_duplicates()


def load_metrics_from_experiment_set(set_prefix: str) -> pd.DataFrame:
    root_dir = REPO_ROOT_PATH / "experiments"
    csv_files = sorted(root_dir.rglob("metrics.csv"))

    all_dfs = []

    for file_path in csv_files:
        if str(file_path.parent.name).startswith(set_prefix):
            df = pd.read_csv(file_path)
            df["folder"] = file_path.parent.name  # keep source info
            all_dfs.append(df)

    concats = pd.concat(all_dfs, ignore_index=True).fillna(0)
    concats.columns = [
        "example_file",
        "Model",
        "Layer",
        "Concept",
        "Similarity",
        "Folder",
    ]
    concats.drop(columns=["example_file"], inplace=True)
    concats = concats[concats["Concept"] != "random"]
    return concats.drop_duplicates()


if __name__ == "__main__":
    """
    WARNING: this script can't yet handle the concept of "Multiple runs".
    You probably need to add it as a dimension to:
     - generated_attributions_on_real_concept
     - similarities
     - true_attributions

    Please have another check at this script before making it go live.
    """
    attributions = load_dataframe_from_experiment_set("3")
    similarities = load_metrics_from_experiment_set("3")

    mask_false_attributions = attributions["Concept"].isin(
        similarities["Concept"].unique()
    )
    true_attributions = attributions[~mask_false_attributions].copy(deep=True)
    true_attributions["Similarity"] = 1
    generated_attributions = attributions[mask_false_attributions]
    generated_attributions_on_real_concept = generated_attributions.drop_duplicates(
        subset=["Model", "Concept", "Layer", "Folder"], keep="first"
    )
    # generated_attributions_on_imputed_concept = generated_attributions.drop_duplicates(subset=['Model', 'Concept', 'Layer', 'Folder'], keep="last")

    index1 = generated_attributions_on_real_concept.set_index(
        ["Model", "Concept", "Layer", "Folder"]
    )
    index2 = similarities.set_index(["Model", "Concept", "Layer", "Folder"])
    joined = index1.join(
        index2, on=["Model", "Concept", "Layer", "Folder"]
    ).reset_index()

    ordered_cols = ["Model", "Layer", "Concept", "Examples"] + [
        col
        for col in joined.columns
        if col not in ["Model", "Layer", "Concept", "Examples"]
    ]
    joined = joined[ordered_cols].drop(columns=["Folder"])
    last_layers = (
        joined.groupby(["Model", "Concept", "Examples"])["Layer"].max().unique()
    )
    plotting_frame = joined[joined["Layer"].isin(last_layers)]
    plotting_frame = plotting_frame.copy()
    plotting_frame['Concept'] = plotting_frame['Concept'].str.split('_bootstrap_').str[0]
    plotting_frame[["mimick", "LLM"]] = plotting_frame["Concept"].str.split(
        "_", n=1, expand=True
    )

    true_attributions = true_attributions.copy()
    true_attributions["mimick"] = true_attributions["Concept"]
    true_attributions["LLM"] = "Original"
    true_attributions = true_attributions[
        true_attributions["Layer"].isin(last_layers)
    ].drop_duplicates(["Model", "Class", "Concept"])

    # Group plotting_frame by Model and mimick
    grouped = plotting_frame.groupby("mimick")

    for cls, class_group in plotting_frame.groupby("Class"):
        model_mimick_combos = class_group[["Model", "mimick"]].drop_duplicates()
        n_combos = len(model_mimick_combos)
        n_cols = math.ceil(math.sqrt(n_combos))
        n_rows = math.ceil(n_combos / n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5), squeeze=False
        )
        fig.suptitle(f"Scatter plots for Class: {cls}", fontsize=16)

        for idx, (model, mimick) in enumerate(model_mimick_combos.values):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]

            group_df = class_group[
                (class_group["Model"] == model) & (class_group["mimick"] == mimick)
            ]
            true_df = true_attributions[
                (true_attributions["mimick"] == mimick)
                & (true_attributions["Model"] == model)
                & (true_attributions["Class"] == cls)
            ]

            # Plot generated points
            sns.scatterplot(
                data=group_df,
                x="Similarity",
                y="Mean",
                hue="LLM",
                palette="tab10",
                edgecolor="black",
                alpha=0.7,
                s=60,
                ax=ax,
            )

            # Overlay true attribution points
            if not true_df.empty:
                sns.scatterplot(
                    data=true_df,
                    x="Similarity",
                    y="Mean",
                    hue="LLM",
                    palette={"Original": "red"},
                    marker="X",
                    s=200,
                    edgecolor="white",
                    linewidth=1.2,
                    ax=ax,
                    legend=False,
                )

            # Aesthetics
            ax.set_xlim(0, 1.1)
            ax.set_ylim(0, max(class_group.Mean.max(), true_df.Mean.max()) + 0.05)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.1)
                spine.set_edgecolor("black")
            ax.set_title(f"Model: {model}\nMimick: {mimick}")
            ax.set_xlabel("Similarity")
            ax.set_ylabel("Mean")

        # Remove unused subplots
        total_plots = n_rows * n_cols
        for empty_idx in range(n_combos, total_plots):
            row, col = divmod(empty_idx, n_cols)
            fig.delaxes(axes[row][col])

        plt.tight_layout(rect=[0, 0, 1, 1])
        LOGGER.log_figure(plt, "attribution_over_similarity")
        plt.show()
