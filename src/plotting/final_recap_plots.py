import argparse

from src.loggers import LOGGER
from src.plotting.plot_data_loader import preprocess_and_clean

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REQUIRED_COLS = [
    "Model","Class","AblationOf","Concept","LLM","Bootstrap",
    "Mean","Std","Similarity"
]

def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    dd = df.copy()
    for c in ["Mean","Std","Similarity","Bootstrap"]:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")
    dd["LLM"] = dd["LLM"].fillna("Real")
    dd["IsAblated"] = dd["AblationOf"].notna() & (dd["AblationOf"].astype(str).str.len() > 0)
    return dd

def _ecdf(y: np.ndarray):
    y = np.asarray(y)
    y = y[~np.isnan(y)]
    if not y.size:
        return np.array([]), np.array([])
    x = np.sort(y)
    f = np.arange(1, x.size + 1) / x.size
    return x, f

def _pair_ablation_deltas(dd: pd.DataFrame) -> pd.DataFrame:
    keys = ["Model","Class","Concept","LLM","Bootstrap"]
    base = dd[~dd["IsAblated"] & (dd["Mean"] > 0.01)][keys+["Mean"]].rename(columns={"Mean":"Mean_nonabl"})
    abl  = dd[dd["IsAblated"] & (dd["AblationOf"] == dd["Concept"])][keys+["Mean"]].rename(columns={"Mean":"Mean_abl"})
    paired = pd.merge(base, abl, on=keys, how="inner")
    paired["Delta_attribution"] = paired["Mean_nonabl"] - paired["Mean_abl"]
    # Compute percentage drop relative to non-ablated attribution
    paired["PctDrop_attribution"] = (
                                            (paired["Mean_nonabl"] - paired["Mean_abl"]) / paired["Mean_nonabl"]
                                    ) * 100.0

    return paired

def _nice_llm_order(dd: pd.DataFrame):
    s = (dd.dropna(subset=["Similarity"])
            .groupby("LLM")["Similarity"].median()
            .sort_values(ascending=False))
    order = list(s.index) if not s.empty else sorted(dd["LLM"].dropna().unique())
    if "Real" in order:
        order.remove("Real"); order = ["Real"] + order
    return order

def _trim_two_sided(arr: np.ndarray, low_q=0.01, high_q=0.99) -> np.ndarray:
    """Return values within [low_q, high_q] quantile range."""
    if arr.size == 0: return arr
    lo, hi = np.nanquantile(arr, [low_q, high_q])
    return arr[(arr >= lo) & (arr <= hi)]

def plot_function(
    df: pd.DataFrame,
    out_dir: str | Path,
    similarity_threshold: float = 0.80,
    # boxplot trimming (set to None to disable)
    trim_q_low: float | None = 0.01,
    trim_q_high: float | None = 0.99,
    # layout
    fig_width: float = 13.5,
    fig_height: float = 8.0,
    dpi: int = 220,
):
    """
    A) ECDF of Similarity (by LLM)
    B) Distribution of global VTCAV Attribution Mean (non-ablated) by LLM
       - Boxplots are trimmed to [trim_q_low, trim_q_high] to avoid outlier-driven squashing.
       - Violins show the *full* data distribution (set `trim_q_*` to None to disable trimming).
    C) Distribution of Δ attribution after ablation (non-abl − abl) by LLM
       - Same trimming for boxplots.
    """
    dd = _prep_df(df)
    paired = _pair_ablation_deltas(dd)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    llm_order = _nice_llm_order(dd)
    cmap = mpl.colormaps.get("tab10")
    colors = {llm: cmap(i % 10) for i, llm in enumerate(llm_order)}

    # layout tuned to avoid overlaps
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, constrained_layout=False)
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.05, 1.0], width_ratios=[1, 1, 1],
        left=0.07, right=0.98, top=0.93, bottom=0.11, wspace=0.40, hspace=0.50
    )
    ax_ecdf  = fig.add_subplot(gs[0, :])
    ax_mean  = fig.add_subplot(gs[1, 0])
    ax_delta = fig.add_subplot(gs[1, 1])
    ax_leg   = fig.add_subplot(gs[1, 2]); ax_leg.axis("off")

    # ---------- A) ECDF ----------
    legend_lines, legend_texts = [], []
    for llm in llm_order:
        sim = dd.loc[dd["LLM"] == llm, "Similarity"].dropna().values
        if sim.size == 0: continue
        x, F = _ecdf(sim)
        ax_ecdf.plot(x, F, lw=2, alpha=0.95, color=colors[llm])
        median = float(np.nanmedian(sim))
        share_ge = float((sim >= similarity_threshold).mean())
        legend_lines.append(Line2D([0],[0], color=colors[llm], lw=3))
        legend_texts.append(f"{llm}: med={median*100:.1f}%, ≥{similarity_threshold*100:.0f}%: {share_ge*100:.0f}%")
    ax_ecdf.set_title("ECDF of CAV Similarity to Real", fontsize=12, pad=3)
    ax_ecdf.set_xlabel("CAV similarity to real (fraction)")
    ax_ecdf.set_ylabel("Empirical CDF")
    ax_ecdf.axhline(0.5, ls="--", lw=1, color="gray", alpha=0.7)
    ax_ecdf.axvline(similarity_threshold, ls="--", lw=1, color="gray", alpha=0.7)
    ax_ecdf.set_xlim(0,1); ax_ecdf.set_ylim(0,1)

    # ---------- B) Mean (non-ablated) ----------
    mean_full, mean_trim, llm_with_mean = [], [], []
    for llm in llm_order:
        arr = dd.loc[(dd["LLM"] == llm) & (~dd["IsAblated"]) & (dd["Mean"] > 0.01), "Mean"].dropna().values
        if arr.size:
            mean_full.append(arr)
            if trim_q_low is None or trim_q_high is None:
                mean_trim.append(arr)
            else:
                mean_trim.append(_trim_two_sided(arr, trim_q_low, trim_q_high))
            llm_with_mean.append(llm)
    if mean_full:
        # violins (full data) — thin and behind to reduce visual overlap
        vp = ax_mean.violinplot(mean_trim, showmeans=False, showmedians=False, showextrema=False)
        for i, b in enumerate(vp["bodies"]):
            b.set_facecolor(colors[llm_with_mean[i]]); b.set_alpha(0.28); b.set_edgecolor("none")
        # boxplots (trimmed) — narrow width to prevent overlaps with violins/each other
        bp = ax_mean.boxplot(
            mean_trim, vert=True, showfliers=False, widths=0.18, patch_artist=True,
            medianprops=dict(linewidth=1.4), whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1)
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[llm_with_mean[i]]); patch.set_alpha(0.65); patch.set_edgecolor("black")
    ax_mean.set_title("Attribution Mean (non-ablated) by LLM", fontsize=12, pad=3)
    ax_mean.set_xlabel("LLM"); ax_mean.set_ylabel("Global attribution (Mean)")
    ax_mean.set_xticks(range(1, len(llm_with_mean)+1))
    ax_mean.set_xticklabels(llm_with_mean, rotation=18, ha="right")

    # ---------- C) % drop (scatter only, ordered by baseline) ----------
    metric = "PctDrop_attribution"
    ax_delta.axhline(0.0, ls="--", lw=1, color="gray", alpha=0.7)

    if not paired.empty:
        pts = paired[["LLM", metric, "Mean_nonabl"]].dropna()
        if not pts.empty:
            llm_to_pos = {llm: i+1 for i, llm in enumerate(llm_order)}

            # Sort so lowest Mean_nonabl plotted first, highest last
            pts = pts.sort_values("Mean_nonabl")

            x = pts["LLM"].map(llm_to_pos).astype(float).values
            x += np.random.normal(0, 0.05, size=x.size)  # jitter
            y = pts[metric].values
            c = pts["Mean_nonabl"].values

            sc = ax_delta.scatter(
                x, y, c=c, cmap="viridis", s=22, alpha=0.6,
                edgecolors="none", zorder=3
            )
            cb = fig.colorbar(sc, ax=ax_delta, fraction=0.045, pad=0.02)
            cb.set_label("Baseline Mean (non-ablated)", rotation=90)

    ax_delta.set_title("% Drop in Attribution after Ablation", fontsize=12, pad=3)
    ax_delta.set_xlabel("LLM")
    ax_delta.set_ylabel("% drop in global attribution")
    ax_delta.set_xticks(range(1, len(llm_order)+1))
    ax_delta.set_xticklabels(llm_order, rotation=18, ha="right")



    # ---------- Legend ----------
    if legend_lines:
        ax_leg.legend(legend_lines, legend_texts, frameon=False, fontsize=10, handlelength=2.5)
        ax_leg.set_title("Similarity @ 50% and ≥ threshold", fontsize=12, pad=2)

    #fig.suptitle("Recap of Visual TCAV Results: Similarity, Attribution Mean, and Ablation Effects", fontsize=14, y=0.965)

    png = Path(out_dir) / "vtcav_recap_mean_trimmed.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return str(png)


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
    plot_function(df=graph_data, out_dir=LOGGER.experiment_path / "images")