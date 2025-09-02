import os
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def _scatter_grid(
    g: pd.DataFrame,
    row_values: list,
    row_mode: str,  # "Ablation" or "concept"
    models: list,
    out_path: str,
    figure_title: str,
    figure_size_per_cell=(4.0, 3.2),
    dpi: int = 220,
):
    """
    Internal helper to render a grid:
      - If row_mode == "Ablation": each row is an AblationOf value (one concept per row)
      - If row_mode == "concept":    each row is a Concept (AblationOf is null)
    Columns are Models in both cases.
    """
    n_rows, n_cols = len(row_values), len(models)
    fig_w = max(1, n_cols) * figure_size_per_cell[0]
    fig_h = max(1, n_rows) * figure_size_per_cell[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)

    # Normalize axes to 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Per-figure limits (+5%)
    x_max = float(np.nanmax(g["Similarity"])) if len(g) else 1.0
    y_max = float(np.nanmax(g["Mean"])) if len(g) else 1.0
    x_lim = (0.0, (x_max * 1.05) if np.isfinite(x_max) else 1.05)
    y_lim = (0.0, (y_max * 1.05) if np.isfinite(y_max) else 1.05)

    # Color map for LLMs (consistent within this figure)
    llms = [l for l in g["LLM"].dropna().unique()]
    cmap = matplotlib.colormaps.get_cmap("tab10")#.resampled(max(1, len(llms)))
    llm_color = {llm: cmap(i) for i, llm in enumerate(llms)}

    for r, row_val in enumerate(row_values):
        if row_mode == "Ablation":
            sub_row = g[g["AblationOf"] == row_val]
            # Exactly one concept per Ablation row
            concept_names = list(sub_row["Concept"].dropna().unique())
            concept_for_title = concept_names[0] if concept_names else "?"
        else:  # "concept"
            concept_for_title = row_val
            sub_row = g[(g["AblationOf"].isna()) & (g["Concept"] == row_val)]

        for c, model in enumerate(models):
            ax = axes[r, c]
            sub = sub_row[sub_row["Model"] == model]

            # Target (LLM null & Bootstrap null) -> red X
            tgt = sub[sub["LLM"].isna() & sub["Bootstrap"].isna()]
            if not tgt.empty:
                ax.scatter(
                    tgt["Similarity"], tgt["Mean"],
                    marker="X", s=80, color="red", label="target"
                )

            # LLM points (possibly multiple bootstraps)
            for llm, grp in sub[~sub["LLM"].isna()].groupby("LLM"):
                ax.scatter(
                    grp["Similarity"], grp["Mean"],
                    s=60, marker="o", color=llm_color.get(llm, "C0"),
                    label=str(llm), edgecolor="black",
                    alpha=0.7
                )

            ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)
            if r == n_rows - 1: ax.set_xlabel("Similarity")
            if c == 0:           ax.set_ylabel("Attribution")

            ax.set_title(f"Model: {model}\nConcept: {concept_for_title}", fontsize=9)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

            # Dedup legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                uniq = dict(zip(labels, handles))
                ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper left", frameon=False)

    fig.suptitle(figure_title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_similarity_vs_attribution_grouped(
    df: pd.DataFrame,
    outdir: str = "figures",
    dpi: int = 220,
    figure_size_per_cell: tuple[float, float] = (4.0, 3.2),
    models_order: Optional[Sequence[str]] = None,
    concepts_order: Optional[Sequence[str]] = None,
):
    """
    Emit up to TWO figures per Class:

      (A) Grouped-by-AblationOf (if there are any non-null AblationOf rows):
            rows = each non-null AblationOf
            cols = Models
          (Each row has exactly one Concept.)

      (B) Old behavior for non-imputed rows (AblationOf is null):
            rows = Concepts (among rows with AblationOf null)
            cols = Models

    Markers: target red 'X'; LLMs colorful circles.
    Limits: per figure, max within figure + 5%.
    """
    req = {'Model','Class','AblationOf','Concept','LLM','Bootstrap','Mean','Std','Similarity'}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    os.makedirs(outdir, exist_ok=True)

    for clazz, gC in df.groupby("Class", dropna=False):
        models = list(dict.fromkeys(models_order or sorted(gC["Model"].unique())))

        # ---- (A) Grouped-by-AblationOf (non-null) ----
        g_imp = gC[gC["AblationOf"].notna()]
        if not g_imp.empty:
            imp_rows = list(dict.fromkeys(sorted(g_imp["AblationOf"].unique())))
            title = f"Similarity vs Attribution — Class: {clazz}\nRows: Imputed Concepts, Cols: Models"
            out_name = os.path.join(outdir, f"scatter_{str(clazz).replace('/','-')}__grouped-by-AblationOf.png")
            _scatter_grid(
                g=g_imp,
                row_values=imp_rows,
                row_mode="Ablation",
                models=models,
                out_path=out_name,
                figure_title=title,
                figure_size_per_cell=figure_size_per_cell,
                dpi=dpi,
            )

        # ---- (B) Old behavior for non-imputed (AblationOf null) ----
        g_nil = gC[gC["AblationOf"].isna()]
        if not g_nil.empty:
            # Rows are the concepts appearing in *non-imputed* rows only
            concepts = list(dict.fromkeys(concepts_order or sorted(g_nil["Concept"].unique())))
            title = f"Similarity vs Attribution — Class: {clazz}\nRows: Concepts, Cols: Models"
            out_name = os.path.join(outdir, f"scatter_{str(clazz).replace('/','-')}__non-imputed_by-Concept.png")
            _scatter_grid(
                g=g_nil,
                row_values=concepts,
                row_mode="concept",
                models=models,
                out_path=out_name,
                figure_title=title,
                figure_size_per_cell=figure_size_per_cell,
                dpi=dpi,
            )
