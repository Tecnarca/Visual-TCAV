import argparse
import os
from typing import Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.loggers import LOGGER
from src.plotting.plot_data_loader import preprocess_and_clean


def plot_strip_points_by_class(
    df: pd.DataFrame,
    outdir: str = "figures",
    y: str = "Mean",  # 'Mean' or 'Similarity'
    dpi: int = 220,
    figure_size_per_cell: tuple[float, float] = (4.2, 3.2),
    models_order: Optional[Sequence[str]] = None,
    concepts_order: Optional[Sequence[str]] = None,
    jitter: float = 0.08,  # horizontal jitter around each category
    s_min: int = 30,       # bumped up for visibility (was 18)
    s_max: int = 100,      # bumped up for visibility (was 60)
    alpha_concept: float = 0.5,   # transparency for filled points
    alpha_imputed: float = 0.5,   # transparency for hollow points' edges
    show_errorbars: bool = False, # set True to add tiny ±Std whiskers
    err_alpha: float = 0.25,      # errorbar transparency (if enabled)
    err_lw: float = 0.7,          # errorbar line width
):
    """
    One figure per Class. Grid of subplots: rows = union(Concept, AblationOf); cols = Models.
    Each subplot uses discrete x-categories: [LLMs..., "Target"].

    Encoding:
    - Color = LLM
    - Filled circle = concept (AblationOf is null)
    - Hollow circle = imputation (AblationOf == concept)
    - Marker size = coefficient of variation CV = Std / (|Mean| + eps) (robustly scaled)
    - Targets share ONE 'Target' column: red filled X (concept) & red hollow X (imputation)
    - Optional tiny errorbars ±Std (off by default to reduce clutter)

    Shared Y-limits per class = max(y)*1.05.

    Note: If y == "Mean", axis labels/titles use the display name "Attribution".
    """
    req = {'Model','Class','AblationOf','Concept','LLM','Bootstrap','Mean','Std','Similarity'}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure numeric
    for col in ["Bootstrap", "Mean", "Std", "Similarity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    os.makedirs(outdir, exist_ok=True)

    def _sanitize(s):
        if pd.isna(s):
            return "None"
        return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))

    def _llm_colors(series: pd.Series):
        uniq = [l for l in series.dropna().unique()]
        cmap = matplotlib.colormaps.get_cmap("tab10")
        return {llm: cmap(i % 10) for i, llm in enumerate(uniq)}

    def _cv_sizes(std: np.ndarray, mean_vals: np.ndarray, s_min: int, s_max: int, p95_cv: float) -> np.ndarray:
        eps = 1e-12
        cv = std / (np.abs(mean_vals) + eps)
        # Robust scaling to [s_min, s_max] using the figure's 95th percentile of CV
        upper = p95_cv if np.isfinite(p95_cv) and p95_cv > 0 else np.nanmax(cv) or 1.0
        upper = upper if upper > 0 else 1.0
        cv = np.clip(cv / upper, 0.0, 1.0)
        return s_min + cv * (s_max - s_min)

    # Use a display-friendly y name without changing which column is plotted
    display_y = "Attribution" if y == "Mean" else y

    for clazz, gC in df.groupby("Class", dropna=False):
        if gC.empty:
            continue

        # Rows (concepts) = union of Concept (non-imputed) and AblationOf (imputed)
        rows_nil = set(gC[gC["AblationOf"].isna()]["Concept"].dropna().unique().tolist())
        rows_imp = set(gC[gC["AblationOf"].notna()]["AblationOf"].dropna().unique().tolist())
        all_rows = rows_nil | rows_imp
        if not all_rows:
            continue

        if concepts_order:
            rows = [c for c in concepts_order if c in all_rows] + [c for c in sorted(all_rows) if c not in set(concepts_order)]
        else:
            rows = sorted(all_rows)

        # Columns (models)
        models = list(dict.fromkeys(models_order or sorted(gC["Model"].unique())))
        if not models:
            continue

        # Y limits per class (+5%)
        y_vals = gC[y].to_numpy(dtype=float)
        y_top = float(np.nanmax(y_vals)) if y_vals.size else 1.0
        y_lim = (0.0, (y_top * 1.05) if np.isfinite(y_top) else 1.05)

        # Colors per LLM
        llm_color = _llm_colors(gC["LLM"])
        category_labels = list(llm_color.keys()) + ["Target"]
        cat_x = np.arange(len(category_labels))

        # Pre-compute a robust CV scale (95th percentile over the class)
        all_cv = (gC["Std"].to_numpy(dtype=float) / (np.abs(gC["Mean"].to_numpy(dtype=float)) + 1e-12))
        p95_cv = float(np.nanpercentile(all_cv[np.isfinite(all_cv)], 95)) if np.any(np.isfinite(all_cv)) else 1.0

        # Layout
        n_rows, n_cols = len(rows), len(models)
        fig_w = n_cols * figure_size_per_cell[0]
        fig_h = n_rows * figure_size_per_cell[1]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        for r, concept_name in enumerate(rows):
            row_nil = gC[(gC["AblationOf"].isna()) & (gC["Concept"] == concept_name)]
            row_imp = gC[(gC["AblationOf"].notna()) & (gC["AblationOf"] == concept_name)]

            for c, model in enumerate(models):
                ax = axes[r, c]
                sub_nil = row_nil[row_nil["Model"] == model]
                sub_imp = row_imp[row_imp["Model"] == model]

                # ----- LLM categories -----
                for llm, col in llm_color.items():
                    x0 = cat_x[list(llm_color.keys()).index(llm)]

                    # Concept (filled)
                    gnil = sub_nil[sub_nil["LLM"] == llm]
                    if not gnil.empty:
                        vals = gnil[y].to_numpy(dtype=float)
                        stds = gnil["Std"].to_numpy(dtype=float)
                        xs = x0 + np.random.uniform(-jitter, jitter, size=len(vals))
                        sizes = _cv_sizes(stds, vals, s_min, s_max, p95_cv)
                        if show_errorbars:
                            ax.errorbar(xs, vals, yerr=stds, fmt='none', ecolor=col, elinewidth=err_lw, alpha=err_alpha, capsize=2)
                        ax.scatter(xs, vals, s=sizes, color=col, alpha=alpha_concept, label=str(llm))

                    # Imputation (hollow)
                    gimp = sub_imp[sub_imp["LLM"] == llm]
                    if not gimp.empty:
                        vals = gimp[y].to_numpy(dtype=float)
                        stds = gimp["Std"].to_numpy(dtype=float)
                        xs = x0 + np.random.uniform(-jitter, jitter, size=len(vals))
                        sizes = _cv_sizes(stds, vals, s_min, s_max, p95_cv)
                        if show_errorbars:
                            ax.errorbar(xs, vals, yerr=stds, fmt='none', ecolor=col, elinewidth=err_lw, alpha=err_alpha, capsize=2)
                        ax.scatter(xs, vals, s=sizes, facecolors=(1,1,1,0), edgecolors=col, linewidths=1.4, alpha=alpha_imputed, label=str(llm) + " (imp)")

                # ----- Targets in shared "Target" category -----
                x_t = cat_x[-1]  # index of "Target"
                tgt_nil = sub_nil[sub_nil["LLM"].isna() & sub_nil["Bootstrap"].isna()]
                if not tgt_nil.empty:
                    xt = x_t + np.random.uniform(-jitter/2, jitter/2, size=len(tgt_nil))
                    ax.scatter(xt, tgt_nil[y], marker="X", s=110, color="red", label="Target (concept)", zorder=3, alpha=0.95)

                tgt_imp = sub_imp[sub_imp["LLM"].isna() & sub_imp["Bootstrap"].isna()]
                if not tgt_imp.empty:
                    xt = x_t + np.random.uniform(-jitter/2, jitter/2, size=len(tgt_imp))
                    ax.scatter(xt, tgt_imp[y], marker="X", s=130, facecolors=(1,1,1,0), edgecolors="red", linewidths=1.6, label="Target (imputation)", zorder=4, alpha=0.95)

                # Cosmetics
                ax.set_ylim(*y_lim)
                ax.set_xlim(-0.5, len(category_labels) - 0.5)
                ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6, axis='y')

                if r == n_rows - 1:
                    ax.set_xticks(cat_x)
                    ax.set_xticklabels(category_labels, rotation=25, ha="right", fontsize=8)
                else:
                    ax.set_xticks([])

                if c == 0:
                    ax.set_ylabel(f"Concept: {concept_name}\n{display_y}", fontsize=9)
                if r == 0:
                    ax.set_title(f"Model: {model}", fontsize=10)

        # ----- Figure-level legend UNDER grid -----
        llm_handles = [Line2D([0], [0], marker='o', linestyle='', color=col, label=str(llm)) for llm, col in llm_color.items()]
        style_handles = [
            Line2D([0], [0], marker='o', linestyle='', color='black', label='Concept', alpha=alpha_concept),
            Line2D([0], [0], marker='o', linestyle='', markerfacecolor='white', markeredgecolor='black', color='black', label='Imputation', alpha=alpha_imputed),
            Line2D([0], [0], marker='X', linestyle='', color='red', label='Target (concept)'),
            Line2D([0], [0], marker='X', linestyle='', markerfacecolor='white', markeredgecolor='red', color='red', label='Target (imputation)'),
        ]
        # Size scale proxies (low/high CV)
        size_handles = [
            Line2D([0], [0], marker='o', linestyle='', color='gray', label='Lower CV', markersize=np.sqrt(s_min)),
            Line2D([0], [0], marker='o', linestyle='', color='gray', label='Higher CV', markersize=np.sqrt(s_max)),
        ]

        handles = llm_handles + style_handles
        labels = [h.get_label() for h in handles]

        fig.suptitle(f"Strip plots over {display_y} — Class: {clazz}", fontsize=12, y=0.98)
        fig.tight_layout(rect=[0, 0.1, 1, 0.94])  # leave bottom space for legend
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=min(6, len(labels)), fontsize=9, frameon=True)

        out_path = os.path.join(outdir, f"strip_{_sanitize(clazz)}_{display_y}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")



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

    # If we have the CSV, generate example plots under /mnt/data/images
    if graph_data is not None:
        out_dir = LOGGER.experiment_path / "images"
        os.makedirs(out_dir, exist_ok=True)
        plot_strip_points_by_class(
            graph_data,
            outdir=LOGGER.experiment_path / "images",
            y="Mean",  # or "Similarity"
            models_order=None,  # or a fixed list to keep column order consistent
            concepts_order=None,  # optionally fix row order
        )

        print(f"Figures saved under {out_dir}")
