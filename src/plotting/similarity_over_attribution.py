import argparse

from src.loggers import LOGGER
from src.plotting.plot_data_loader import preprocess_and_clean
from src.plotting.plotter import plot_similarity_vs_attribution_grouped

import os
import math
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- Utility used by both grouped and combined modes ----
def _sanitize(s):
    if pd.isna(s):
        return "None"
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))

def _llm_colors(series: pd.Series):
    """Stable color mapping for LLMs inside a figure."""
    llms = [l for l in series.dropna().unique()]
    cmap = matplotlib.colormaps.get_cmap("tab10")
    return {llm: cmap(i % 10) for i, llm in enumerate(llms)}

# =========================
#   NEW: Combined mode
# =========================
def plot_similarity_vs_attribution_combined(
    df: pd.DataFrame,
    outdir: str = "figures",
    dpi: int = 220,
    figure_size_per_cell: tuple[float, float] = (4.0, 3.2),
    models_order: Optional[Sequence[str]] = None,
    concepts_order: Optional[Sequence[str]] = None,
):
    """
    Single image per Class combining:
      (A) Imputed rows (AblationOf non-null) and
      (B) Non-imputed rows (AblationOf null)
    Rows are the union of concept names from (B) and AblationOf names from (A).
    If both exist for the same concept name, they're plotted on the SAME subplot:
      - Concept points/X:   filled markers
      - Imputation points:  hollow (facecolor='white', colored edge)
      - Imputation X:       'X' with facecolor='white', colored edge

    Columns are Models. Axes limits are per-figure, set to max + 5%.
    """
    req = {'Model','Class','AblationOf','Concept','LLM','Bootstrap','Mean','Std','Similarity'}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for combined mode: {sorted(missing)}")

    # Ensure numerics
    for col in ["Bootstrap", "Mean", "Std", "Similarity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    os.makedirs(outdir, exist_ok=True)

    for clazz, gC in df.groupby("Class", dropna=False):
        if gC.empty:
            continue

        # Figure-wide limits (+5%)
        x_max = float(np.nanmax(gC["Similarity"])) if len(gC) else 1.0
        y_max = float(np.nanmax(gC["Mean"])) if len(gC) else 1.0
        x_lim = (0.0, (x_max * 1.05) if np.isfinite(x_max) else 1.05)
        y_lim = (0.0, (y_max * 1.05) if np.isfinite(y_max) else 1.05)

        # Split imputed vs non-imputed
        g_imp = gC[gC["AblationOf"].notna()].copy()
        g_nil = gC[gC["AblationOf"].isna()].copy()

        # Rows: union of names from AblationOf and Concept
        rows_imp = set(g_imp["AblationOf"].dropna().unique().tolist())
        rows_nil = set(g_nil["Concept"].dropna().unique().tolist())
        all_rows = rows_imp | rows_nil

        # Optional external ordering
        if concepts_order:
            # Keep given order and then append any missing at the end
            ordered = [c for c in concepts_order if c in all_rows] + \
                      [c for c in sorted(all_rows) if c not in set(concepts_order)]
        else:
            ordered = sorted(all_rows)

        # Columns: models
        models = list(dict.fromkeys(models_order or sorted(gC["Model"].unique())))

        n_rows, n_cols = len(ordered), len(models)
        if n_rows == 0 or n_cols == 0:
            continue

        fig_w = n_cols * figure_size_per_cell[0]
        fig_h = n_rows * figure_size_per_cell[1]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)

        # Normalize axes to 2D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        # LLM color map consistent within figure
        llm_color = _llm_colors(gC["LLM"])

        for r, concept_name in enumerate(ordered):
            # Data for this row
            row_nil = g_nil[g_nil["Concept"] == concept_name]       # filled
            row_imp = g_imp[g_imp["AblationOf"] == concept_name]    # hollow

            # Derive a human title for the row (prefer explicit concept name)
            row_title = str(concept_name)

            for c, model in enumerate(models):
                ax = axes[r, c]

                sub_nil = row_nil[row_nil["Model"] == model]
                sub_imp = row_imp[row_imp["Model"] == model]

                # ---- PLOT NON-IMPUTED (filled) ----
                # Target concept (LLM null & Bootstrap null)
                tgt_nil = sub_nil[sub_nil["LLM"].isna() & sub_nil["Bootstrap"].isna()]
                if not tgt_nil.empty:
                    ax.scatter(
                        tgt_nil["Similarity"], tgt_nil["Mean"],
                        marker="X", s=90, color="red", label="target (concept)", zorder=3
                    )
                # LLM points
                for llm, grp in sub_nil[~sub_nil["LLM"].isna()].groupby("LLM"):
                    col = llm_color.get(llm, "C0")
                    ax.scatter(
                        grp["Similarity"], grp["Mean"],
                        s=65, marker="o", color=col, edgecolor="black", linewidths=0.7,
                        alpha=0.8, label=f"{llm} (concept)"
                    )

                # ---- PLOT IMPUTED (hollow) ----
                tgt_imp = sub_imp[sub_imp["LLM"].isna() & sub_imp["Bootstrap"].isna()]
                if not tgt_imp.empty:
                    ax.scatter(
                        tgt_imp["Similarity"], tgt_imp["Mean"],
                        marker="X", s=110, facecolor="white", edgecolor="red",
                        linewidths=1.6, label="target (imputation)", zorder=4
                    )
                for llm, grp in sub_imp[~sub_imp["LLM"].isna()].groupby("LLM"):
                    col = llm_color.get(llm, "C0")
                    ax.scatter(
                        grp["Similarity"], grp["Mean"],
                        s=65, marker="o", facecolor="white", edgecolor=col,
                        linewidths=1.4, alpha=0.95, label=f"{llm} (imputation)"
                    )

                # Axes, title, grid
                ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)
                if r == n_rows - 1: ax.set_xlabel("Similarity")
                if c == 0:           ax.set_ylabel("Attribution")
                ax.set_title(f"Model: {model}\nConcept: {row_title}", fontsize=9)
                ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

                # De-duplicate labels per-axes
                h, l = ax.get_legend_handles_labels()
                uniq = {}
                for handle, label in zip(h, l):
                    if label and label not in uniq:
                        uniq[label] = handle
                # Keep legend small on axes; we’ll add a figure-level style legend too
                if uniq:
                    ax.legend(list(uniq.values())[:3], list(uniq.keys())[:3], fontsize=7,
                              loc="upper left", frameon=False)

        # Figure title
        fig.suptitle(
            f"Similarity vs Attribution — Class: {clazz}\nRows: Concepts ∪ Imputed (hollow)",
            fontsize=12, y=1.02
        )

        # ---- Figure-level STYLE legend (explains filled vs hollow once) ----
        proxy = [
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor='black', markeredgecolor='black', label='Concept point (filled)'),
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor='white', markeredgecolor='black', label='Imputation point (hollow)'),
            Line2D([0], [0], marker='X', linestyle='', color='red', markersize=9,
                   label='Target (concept)'),
            Line2D([0], [0], marker='X', linestyle='', markerfacecolor='white',
                   markeredgecolor='red', markersize=10, label='Target (imputation)')
        ]
        fig.legend(handles=proxy, loc="upper center", ncol=2, fontsize=9, frameon=True)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_name = os.path.join(outdir, f"scatter_{_sanitize(clazz)}__combined.png")
        fig.savefig(out_name, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_experiment_set",
        type=str,
        default="test_runs_202508",
        help="Prefix of the experiment set to load (default: test_runs)"
    )
    args = parser.parse_args()

    graph_data = preprocess_and_clean(args.from_experiment_set)

    plot_similarity_vs_attribution_grouped(graph_data, LOGGER.experiment_path / "images")

    plot_similarity_vs_attribution_combined(
        graph_data,
        outdir=LOGGER.experiment_path / "images",
        models_order=None,  # or a list like ["resnet50", "vit-b16", ...]
        concepts_order=None,  # or a preferred row ordering
        dpi=220,
        figure_size_per_cell=(4.0, 3.2),
    )
