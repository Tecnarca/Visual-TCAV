#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

# ======== Config ========
MODELS: List[str] = ["flux", "sd35"]
#MODELS: List[str] = ["gpti1"]
TARGET_PER_CONCEPT = 200
FNAME_RE_CACHE = {}  # cache compiled regex per model
# ========================

# ===== NEW: estimation helpers =====
from collections import defaultdict

def _have_count_like_runtime(target_dir: Path, model: str) -> int:
    """
    Mirrors the runtime's notion of 'already' images:
    the maximum index among files named {model}_{N}.png (not the count).
    """
    return _max_existing_index(target_dir, model)

def estimate_missing(df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    """
    Build a per-(Concept, Model) estimate with columns:
    Concept, Model, Have, Need (Need = max(0, TARGET_PER_CONCEPT - Have)).
    """
    rows = []
    for _, row in df.iterrows():
        concept = str(row["Concept"])
        for model in MODELS:
            target_dir = output_root / f"{concept}_{model}"
            have = _have_count_like_runtime(target_dir, model)
            need = max(0, TARGET_PER_CONCEPT - have)
            rows.append({"Concept": concept, "Model": model, "Have": have, "Need": need})
    return pd.DataFrame(rows)

def print_estimate_summary(est_df: pd.DataFrame):
    """
    Pretty-print a compact summary:
    - Totals per model
    - Optional head of per-concept breakdown (sorted by 'Need' desc)
    """
    if est_df.empty:
        print("[estimate] No concepts to estimate.")
        return

    # Totals per model
    model_totals = (est_df.groupby("Model")["Need"]
                    .sum()
                    .sort_values(ascending=False))
    print("\n====== Missing Images: Totals per Model ======")
    for model, need_sum in model_totals.items():
        print(f"{model:>12}: {need_sum}")

    # Overall totals
    grand_total = int(model_totals.sum())
    concepts = est_df["Concept"].nunique()
    print("==============================================")
    print(f" Concepts: {concepts}")
    print(f"   Target: {TARGET_PER_CONCEPT} per (Concept, Model)")
    print(f"   Missing (grand total): {grand_total}")

    # Per-concept top needs (optional, helpful when many concepts)
    print("\n-- Top missing (by Concept, per Model) --")
    top = (est_df.sort_values(["Need", "Concept", "Model"], ascending=[False, True, True])
                 .query("Need > 0"))
    # Limit to a reasonable preview if large
    preview_rows = min(25, len(top))
    if preview_rows:
        print(top.head(preview_rows).to_string(index=False))
    else:
        print("All complete 🎉")
# ===== END new estimation helpers =====


@dataclass
class GenTask:
    concept: str
    positive: str
    negative: str

def _compiled_name_re(model: str):
    """Match files like 'flux_1.png' or 'stable_diffusion_42.png' or 'gpt-image-1_200.png'."""
    if model not in FNAME_RE_CACHE:
        FNAME_RE_CACHE[model] = re.compile(rf"^{re.escape(model)}_(\d+)\.png$")
    return FNAME_RE_CACHE[model]


def _max_existing_index(target_dir: Path, model: str) -> int:
    """Return max N among files named {model}_{N}.png in target_dir."""
    if not target_dir.exists():
        return 0
    pat = _compiled_name_re(model)
    max_idx = 0
    for p in target_dir.glob("*.png"):
        m = pat.match(p.name)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
    return max_idx


def _move_and_renumber(staging_dir: Path, target_dir: Path, model: str, start_index: int) -> int:
    """Move PNGs from staging_dir to target_dir as {model}_{start+1..}.png; return count moved."""
    target_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted([p for p in staging_dir.glob("*.png") if p.is_file()])
    n = 0
    curr = start_index
    for src in pngs:
        curr += 1
        dst = target_dir / f"{model}_{curr}.png"
        while dst.exists():
            curr += 1
            dst = target_dir / f"{model}_{curr}.png"
        shutil.move(str(src), str(dst))
        n += 1
    return n


def _clean_dir(path: Path):
    if not path.exists():
        return
    for item in path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        path.rmdir()
    except Exception:
        pass


def generate_from_df(
    script_path: Path,
    df: pd.DataFrame,
    output_root: Path = Path("generated_images"),
    staging_root: Path = Path(".staging_generations"),
):
    """
    For each Concept in df, generate 200 images with each model.
    Uses columns: Concept, Positive, Negative (Class ignored).
    Output: generated_images/{concept}_{model}/{model}_1.png..{model}_200.png
    Resumable.
    """
    required_cols = {"Concept", "Positive", "Negative"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Concept"] = df["Concept"].astype(str)
    df["Positive"] = df["Positive"].astype(str)
    df["Negative"] = df["Negative"].astype(str)

    total_batches = len(df) * len(MODELS)
    pbar = tqdm(total=total_batches, desc="Concept x Model batches", unit="batch")

    for _, row in df.iterrows():
        concept = row["Concept"]
        pos_prompt = row["Positive"]
        neg_prompt = row["Negative"]

        for model in MODELS:
            target_dir = output_root / f"{concept}_{model}"
            current_max = _max_existing_index(target_dir, model)
            already = current_max
            remaining = max(0, TARGET_PER_CONCEPT - already)

            if remaining <= 0:
                tqdm.write(f"[skip] Concept='{concept}' Model='{model}' already has {TARGET_PER_CONCEPT} images.")
                pbar.update(1)
                continue

            tqdm.write(
                f"[run] Concept='{concept}' Model='{model}' -> have={already}, need={remaining}, out={target_dir}"
            )

            stage_dir = staging_root / f"{concept}_{model}"
            _clean_dir(stage_dir)
            stage_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(script_path),
                "--prompt", pos_prompt,
                "--model", model,
                "--num_images", str(remaining),
                "--output_dir", str(stage_dir),
            ]
            if neg_prompt.strip():
                cmd.extend(["--negative_prompt", neg_prompt])

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                tqdm.write(
                    f"[error] Generation failed for Concept='{concept}' Model='{model}' "
                    f"(exit {e.returncode}). Keeping partial results; will resume next run."
                )
            except Exception as e:
                tqdm.write(
                    f"[error] Generation failed for Concept='{concept}' Model='{model}': {e}. "
                    "Keeping partial results; will resume next run."
                )

            moved = _move_and_renumber(stage_dir, target_dir, model, start_index=current_max)
            _clean_dir(stage_dir)

            tqdm.write(
                f"[done] Concept='{concept}' Model='{model}' moved {moved} files -> {target_dir}. "
                f"Total now: {already + moved}/{TARGET_PER_CONCEPT}"
            )

            pbar.update(1)

    pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Generate images for each concept across all models (resumable).")
    parser.add_argument(
        "--script_path",
        type=Path,
        default=Path(__file__).parent / "generative_pipelines.py",
        help="Path to your existing image script (with ImageGenerator CLI).",
    )
    parser.add_argument(
        "--df_csv",
        type=Path,
        default=Path(__file__).parent / "concept_prompts.csv",
        help="CSV with columns: Concept, Positive, Negative. (Class is ignored.)",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("generated_images"),
        help="Root output directory (default: generated_images).",
    )
    parser.add_argument(
        "--staging_root",
        type=Path,
        default=Path(".staging_generations"),
        help="Temporary staging directory (default: .staging_generations).",
    )
    args = parser.parse_args()

    # 3) Load DF and generate
    df = pd.read_csv(args.df_csv)

    est_df = estimate_missing(df=df, output_root=args.output_root)
    print_estimate_summary(est_df)

    generate_from_df(
        script_path=args.script_path,
        df=df,
        output_root=args.output_root,
        staging_root=args.staging_root,
    )


if __name__ == "__main__":
    main()
