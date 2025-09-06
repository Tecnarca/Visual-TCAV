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
    paired["PctDrop_attribution"] = (
        (paired["Mean_nonabl"] - paired["Mean_abl"]) / paired["Mean_nonabl"]
    ) * 100.0
    return paired

def _llm_vs_real_ablated_diffs(dd: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Compute (Mean_llm - Mean_real) for ablated items.
    We only keep rows where the ablation matches the concept (AblationOf == Concept),
    group by (Model, Class, Concept, LLM), pivot to columns per LLM, and subtract Real.
    """
    ablated = dd[(dd["IsAblated"]) & (dd["AblationOf"] == dd["Concept"]) & (dd["Mean"] > 0.0)].copy()
    if ablated.empty:
        return {}

    grouped = (ablated
        .groupby(["Model","Class","Concept","LLM"], dropna=False)["Mean"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot_table(index=["Model","Class","Concept"], columns="LLM", values="Mean", aggfunc="mean")
    if "Real" not in pivot.columns:
        return {}

    diffs: dict[str, np.ndarray] = {}
    real_vals = pivot["Real"]
    for llm in pivot.columns:
        if llm == "Real":
            continue
        diffs[llm] = (pivot[llm] - real_vals).dropna().to_numpy()
    return diffs


def _llm_vs_real_nonablated_diffs(dd: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Robustly compute (Mean_llm - Mean_real) for non-ablated items.
    We group by (Model, Class, Concept, LLM), average across any Bootstraps,
    pivot to columns per LLM, and subtract the Real column.
    """
    nonabl = dd[(~dd["IsAblated"]) & (dd["Mean"] > 0.01)].copy()
    if nonabl.empty:
        return {}

    grouped = (nonabl
        .groupby(["Model","Class","Concept","LLM"], dropna=False)["Mean"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot_table(index=["Model","Class","Concept"], columns="LLM", values="Mean", aggfunc="mean")
    if "Real" not in pivot.columns:
        return {}

    diffs: dict[str, np.ndarray] = {}
    real_vals = pivot["Real"]
    for llm in pivot.columns:
        if llm == "Real":
            # Difference of Real to itself -> zeros (useful for completeness)
            diffs[llm] = np.zeros_like(real_vals.to_numpy(), dtype=float)
            continue
        arr = (pivot[llm] - real_vals).dropna().to_numpy()
        diffs[llm] = arr
    return diffs

def _nice_llm_order(dd: pd.DataFrame):
    s = (dd.dropna(subset=["Similarity"])
            .groupby("LLM")["Similarity"].median()
            .sort_values(ascending=False))
    order = list(s.index) if not s.empty else sorted(dd["LLM"].dropna().unique())
    if "Real" in order:
        order.remove("Real"); order = ["Real"] + order
    return order

def _trim_two_sided(arr: np.ndarray, low_q=0.01, high_q=0.99) -> np.ndarray:
    if arr.size == 0: return arr
    lo, hi = np.nanquantile(arr, [low_q, high_q])
    return arr[(arr >= lo) & (arr <= hi)]
def plot_function(
    df: pd.DataFrame,
    out_dir: str | Path,
    similarity_threshold: float = 0.80,
    trim_q_low: float | None = 0.01,
    trim_q_high: float | None = 0.99,
    fig_width: float = 13.5,
    fig_height: float = 8.8,
    dpi: int = 220,
):
    dd = _prep_df(df)
    paired = _pair_ablation_deltas(dd)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    llm_order = _nice_llm_order(dd)
    cmap = mpl.colormaps.get("tab10")
    colors = {llm: cmap(i % 10) for i, llm in enumerate(llm_order)}

    # === Layout: A (top-left), C (top-right), B (bottom-left), D (bottom-right) ===
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, constrained_layout=False)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.05, 1.0], width_ratios=[1.35, 1.0],
        left=0.07, right=0.98, top=0.93, bottom=0.12, wspace=0.35, hspace=0.58
    )

    # A: ECDF (top-left)
    ax_ecdf  = fig.add_subplot(gs[0, 0])
    # C: 2×2 small multiples (top-right)
    subgs_top_right = gs[0, 1].subgridspec(2, 2, wspace=0.18, hspace=0.30)
    # B: Δ Mean (non-ablated) (bottom-left)
    ax_mean_nonabl = fig.add_subplot(gs[1, 0])
    # D: Δ Mean (ablated) (bottom-right)
    ax_mean_abl    = fig.add_subplot(gs[1, 1])

    # ---------- A) ECDF (with median markers) ----------
    legend_lines, legend_texts = [], []
    for llm in llm_order:
        sim = dd.loc[dd["LLM"] == llm, "Similarity"].dropna().values
        if sim.size == 0:
            continue
        x, F = _ecdf(sim)
        ax_ecdf.plot(x, F, lw=2.4, alpha=0.95, color=colors[llm], zorder=2)

        median = float(np.nanmedian(sim))
        ax_ecdf.plot([median], [0.5], marker="o", ms=5.5, color=colors[llm], zorder=3)
        share_ge = float((sim >= similarity_threshold).mean())

        legend_lines.append(Line2D([0],[0], color=colors[llm], lw=3))
        legend_texts.append(f"{llm}: med={median*100:.1f}%, ≥{similarity_threshold*100:.0f}%: {share_ge*100:.0f}%")

    ax_ecdf.set_title("ECDF of CAV Similarity to Real", fontsize=12, pad=3)
    ax_ecdf.set_xlabel("CAV similarity to real (fraction)")
    ax_ecdf.set_ylabel("Empirical CDF")
    ax_ecdf.axhline(0.5, ls="--", lw=1, color="gray", alpha=0.7, zorder=1)
    ax_ecdf.axvline(similarity_threshold, ls="--", lw=1, color="gray", alpha=0.7, zorder=1)
    ax_ecdf.set_xlim(0,1); ax_ecdf.set_ylim(0,1)
    if legend_lines:
        ax_ecdf.legend(legend_lines, legend_texts, frameon=False, fontsize=10,
                       loc="lower right", title="Similarity @ 50% and ≥ threshold")

    # ---------- C) Top-right: 2×2 small multiples ----------
    metric = "PctDrop_attribution"
    pts = paired[["LLM", metric, "Mean_nonabl"]].dropna()

    llms_panels = llm_order[:4] if len(llm_order) >= 4 else llm_order

    # shared limits across panels
    x_all, y_all = [], []
    for llm in llms_panels:
        sub = pts.loc[pts["LLM"] == llm]
        if not sub.empty:
            x_all.append(sub["Mean_nonabl"].to_numpy())
            y_all.append(sub[metric].to_numpy())
    x_all = np.concatenate(x_all) if x_all else np.array([])
    y_all = np.concatenate(y_all) if y_all else np.array([])
    x_min, x_max = (np.nanmin(x_all), np.nanmax(x_all)) if x_all.size else (0.0, 1.0)
    y_min, y_max = (np.nanmin(y_all), np.nanmax(y_all)) if y_all.size else (-100.0, 100.0)
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.05
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 5.0
    xlim = (max(0.0, x_min - x_pad), min(1.0, x_max + x_pad))
    ylim = (y_min - y_pad, y_max + y_pad)

    axes_small = []
    for idx, llm in enumerate(llms_panels):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(subgs_top_right[r, c]); axes_small.append(ax)

        sub = pts.loc[pts["LLM"] == llm]
        if not sub.empty:
            ax.scatter(sub["Mean_nonabl"].to_numpy(), sub[metric].to_numpy(),
                       s=22, alpha=0.6, edgecolors="none", color=colors.get(llm, "C0"))
        ax.axhline(0.0, ls="--", lw=1, color="gray", alpha=0.7)
        ax.set_title(llm, fontsize=11, pad=2)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(""); ax.set_ylabel("")

    # shared labels + block title for the 2×2
    bbox_left   = min(ax.get_position().x0 for ax in axes_small)
    bbox_right  = max(ax.get_position().x1 for ax in axes_small)
    bbox_bottom = min(ax.get_position().y0 for ax in axes_small)
    bbox_top    = max(ax.get_position().y1 for ax in axes_small)
    mid_x = (bbox_left + bbox_right) / 2.0
    mid_y = (bbox_bottom + bbox_top) / 2.0

    fig.text(bbox_left - 0.035, mid_y, "% drop in attribution",
             rotation=90, va="center", ha="right")
    fig.text(mid_x, bbox_bottom - 0.03, "Non-ablated Attribution",
             va="top", ha="center")
    fig.text(mid_x, min(0.985, bbox_top + 0.02),
             "Ablation effect by LLM: % drop in attribution",
             ha="center", va="bottom", fontsize=11)

    # ---------- B) Bottom-left: Δ Mean (non-ablated) vs Real ----------
    diffs_nonabl = _llm_vs_real_nonablated_diffs(dd)
    series_nonabl, labels_nonabl = [], []
    for llm in [x for x in llm_order if x != "Real"]:
        arr = diffs_nonabl.get(llm, np.array([]))
        if arr.size:
            arr = _trim_two_sided(arr, trim_q_low, trim_q_high) if (trim_q_low is not None and trim_q_high is not None) else arr
            series_nonabl.append(arr); labels_nonabl.append(llm)

    if series_nonabl:
        vp = ax_mean_nonabl.violinplot(series_nonabl, showmeans=False, showmedians=False, showextrema=False)
        for i, b in enumerate(vp["bodies"]):
            b.set_facecolor(colors[labels_nonabl[i]]); b.set_alpha(0.28); b.set_edgecolor("none")
        bp = ax_mean_nonabl.boxplot(
            series_nonabl, vert=True, showfliers=False, widths=0.18, patch_artist=True,
            medianprops=dict(linewidth=1.4), whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1)
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[labels_nonabl[i]]); patch.set_alpha(0.65); patch.set_edgecolor("black")

    ax_mean_nonabl.axhline(0.0, ls="--", lw=1, color="gray", alpha=0.7)
    ax_mean_nonabl.set_title("Δ Mean (non-ablated) vs Real", fontsize=12, pad=3)
    ax_mean_nonabl.set_xlabel("LLM")
    ax_mean_nonabl.set_ylabel("Attribution: LLM − Real")
    ax_mean_nonabl.set_xticks(range(1, len(labels_nonabl)+1))
    ax_mean_nonabl.set_xticklabels(labels_nonabl, rotation=18, ha="right")

    # ---------- D) Bottom-right: Δ Mean (ablated) vs Real ----------
    diffs_abl = _llm_vs_real_ablated_diffs(dd)
    series_abl, labels_abl = [], []
    for llm in [x for x in llm_order if x != "Real"]:
        arr = diffs_abl.get(llm, np.array([]))
        if arr.size:
            arr = _trim_two_sided(arr, trim_q_low, trim_q_high) if (trim_q_low is not None and trim_q_high is not None) else arr
            series_abl.append(arr); labels_abl.append(llm)

    if series_abl:
        vp = ax_mean_abl.violinplot(series_abl, showmeans=False, showmedians=False, showextrema=False)
        for i, b in enumerate(vp["bodies"]):
            b.set_facecolor(colors[labels_abl[i]]); b.set_alpha(0.28); b.set_edgecolor("none")
        bp = ax_mean_abl.boxplot(
            series_abl, vert=True, showfliers=False, widths=0.18, patch_artist=True,
            medianprops=dict(linewidth=1.4), whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1)
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[labels_abl[i]]); patch.set_alpha(0.65); patch.set_edgecolor("black")

    ax_mean_abl.axhline(0.0, ls="--", lw=1, color="gray", alpha=0.7)
    ax_mean_abl.set_title("Δ Mean (ablated) vs Real", fontsize=12, pad=3)
    ax_mean_abl.set_xlabel("LLM")
    ax_mean_abl.set_ylabel("Attribution (ablation): LLM − Real")
    ax_mean_abl.set_xticks(range(1, len(labels_abl)+1))
    ax_mean_abl.set_xticklabels(labels_abl, rotation=18, ha="right")

    # ---------- Write similarity summary to CSV ----------
    summary_rows = []
    for llm in llm_order:
        sim = dd.loc[dd["LLM"] == llm, "Similarity"].dropna().values
        if sim.size == 0:
            summary_rows.append(
                {"LLM": llm, "mean": np.nan, "p0": np.nan, "p50": np.nan, "p80": np.nan, "p100": np.nan}
            )
            continue
        summary_rows.append(
            {
                "LLM": llm,
                "mean": float(np.nanmean(sim)),
                "p0":   float(np.nanpercentile(sim, 0)),
                "p50":  float(np.nanpercentile(sim, 50)),
                "p80":  float(np.nanpercentile(sim, 80)),
                "p100": float(np.nanpercentile(sim, 100)),
            }
        )

    summary_df = pd.DataFrame(summary_rows, columns=["LLM","mean","p0","p50","p80","p100"])
    csv_path = Path(out_dir) / "similarity_summary.csv"
    summary_df.to_csv(csv_path, index=False)


    # Save
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