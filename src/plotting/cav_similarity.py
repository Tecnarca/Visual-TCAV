import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.loggers import REPO_ROOT_PATH, LOGGER

def split_prefix_suffix(concept_name):
    if "_" in concept_name:
        prefix, suffix = concept_name.split("_", 1)
        if "_" in suffix:
            suffix = suffix.split("_", 1)[0]
    else:
        prefix, suffix = "unknown", concept_name
    return prefix, suffix

if __name__ == "__main__":
    split_by_key0 = False
    root_dir = REPO_ROOT_PATH / "experiments"
    csv_files = sorted(root_dir.rglob("metrics.csv"))

    all_dfs = []

    for file_path in csv_files:
        if str(file_path.parent.name).startswith("2"):
            df = pd.read_csv(file_path)
            df["source"] = file_path.parent.name  # keep source info
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid CSV files found.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["value"]).copy()
    df_all[["prefix", "suffix"]] = df_all["key_2"].apply(lambda x: pd.Series(split_prefix_suffix(x)))
    df_all = df_all[df_all["prefix"] != "unknown"]

    last_layers = df_all.groupby('key0')['key_1'].max().unique()
    df_all = df_all[df_all["key_1"].isin(last_layers)]

    import math

    # Group the data
    grouped = df_all.groupby(["key0", "prefix", "suffix"])["value"].agg(["mean", "std"]).fillna(0)
    grouped["std"] *= 2

    if split_by_key0:
        # -------- GRID OF SUBPLOTS (one per key0) --------
        import math

        key0_categories = grouped.index.get_level_values("key0").unique()
        prefixes = grouped.index.get_level_values("prefix").unique()
        suffixes = grouped.index.get_level_values("suffix").unique()

        n_plots = len(key0_categories)
        n_cols = math.ceil(math.sqrt(n_plots))
        n_rows = math.ceil(n_plots / n_cols)

        x = np.arange(len(suffixes))
        total_width = 0.8
        bar_width = total_width / len(prefixes)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)
        axes = axes.flatten()

        for idx, key0 in enumerate(key0_categories):
            ax = axes[idx]

            for i, prefix in enumerate(prefixes):
                means = []
                stds = []
                for suffix in suffixes:
                    key = (key0, prefix, suffix)
                    if key in grouped.index:
                        means.append(grouped.loc[key, "mean"])
                        stds.append(grouped.loc[key, "std"])
                    else:
                        means.append(0)
                        stds.append(0)

                bar_positions = x + i * bar_width - total_width / 2 + bar_width / 2

                ax.bar(
                    bar_positions,
                    means,
                    width=bar_width,
                    yerr=stds,
                    capsize=4,
                    label=prefix,
                    ecolor="black",
                    error_kw=dict(lw=1, ls="--"),
                )

            ax.set_title(f"{key0}")
            ax.set_xticks(x)
            ax.set_xticklabels(suffixes, rotation=45)
            ax.set_ylim(0, 1)
            if idx % n_cols == 0:
                ax.set_ylabel("Cosine similarity")
            if idx >= (n_rows - 1) * n_cols:
                ax.set_xlabel("LLM Model")

        for i in range(n_plots, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle("CAVs over concepts and LLM models (Mean ± 2*Std)", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title="Concept (prefix)", loc="upper right")

        LOGGER.log_figure(plt, "cav_analysis_grid")
        plt.show()

    else:
        # -------- AGGREGATED SINGLE PLOT --------
        grouped_agg = df_all.groupby(["prefix", "suffix"])["value"].agg(["mean", "std"]).fillna(0)
        grouped_agg["std"] *= 2

        prefixes = grouped_agg.index.get_level_values("prefix").unique()
        suffixes = grouped_agg.index.get_level_values("suffix").unique()

        x = np.arange(len(suffixes))
        total_width = 0.8
        bar_width = total_width / len(prefixes)

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, prefix in enumerate(prefixes):
            means = []
            stds = []
            for suffix in suffixes:
                key = (prefix, suffix)
                if key in grouped_agg.index:
                    means.append(grouped_agg.loc[key, "mean"])
                    stds.append(grouped_agg.loc[key, "std"])
                else:
                    means.append(0)
                    stds.append(0)

            bar_positions = x + i * bar_width - total_width / 2 + bar_width / 2

            ax.bar(
                bar_positions,
                means,
                width=bar_width,
                yerr=stds,
                capsize=4,
                label=prefix,
                ecolor="black",
                error_kw=dict(lw=1, ls="--"),
            )

        ax.set_title("CAVs over concepts and LLM models (Aggregated, Mean ± 2*Std)")
        ax.set_ylabel("Cosine similarity to ground truth")
        ax.set_xlabel("LLM Model")
        ax.set_xticks(x)
        ax.set_xticklabels(suffixes, rotation=45)
        ax.set_ylim(0, 1)
        ax.legend(title="Concept (prefix)")
        plt.tight_layout()

        LOGGER.log_figure(plt, "cav_analysis_aggregated")
        plt.show()
