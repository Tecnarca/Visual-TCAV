import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Union

import pandas as pd
from tqdm import tqdm


def run_concept_stripping_from_df_batched(
    script_path: Union[str, Path],
    df: pd.DataFrame,
    classes_root: Union[
        str, Path
    ] = "/home/tecnarca/PycharmProjects/Visual-TCAV/VisualTCAV/test_images",
    output_root: Union[str, Path] = "edited_images",
    staging_root: Union[str, Path] = ".staging_edits",
    exts: Iterable[str] = (".png", ".jpg", ".jpeg"),
) -> None:
    """
    Batch-optimized: call the image-edit script ONCE per (Class, Concept), feeding a folder
    that contains symlinks ONLY to images that still need editing. This preserves resumability
    while minimizing subprocess overhead.

    DF schema: columns ['Class', 'Concept'].
    Output folder: edited_images/<Class>_un<Concept>/

    Notes:
    - Uses symlinks in a per-(Class,Concept) staging folder so the script receives a directory.
    - If the OS/filesystem doesn't allow symlinks, you can swap to hardlinks via os.link.
    """
    script_path = Path(script_path)
    classes_root = Path(classes_root)
    output_root = Path(output_root)
    staging_root = Path(staging_root)

    required_cols = {"Class", "Concept"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Class"] = df["Class"].astype(str)
    df["Concept"] = df["Concept"].astype(str)

    # Global progress over all (Class, Concept) pairs
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Concept batches", unit="batch"
    ):
        cls = row["Class"]
        concept = row["Concept"]

        class_dir = classes_root / cls
        if not class_dir.is_dir():
            tqdm.write(f"[skip] Class folder missing: {class_dir}")
            continue

        # Build prompt (underscores -> spaces only in the text)
        c_text = concept.replace("_", " ")
        prompt = f"""
            Completely remove the concept of {c_text} from the image, 
            keeping it as visually similar possible without {c_text} being present at all. 
            """

        # Output dir name: f"{class}_un{concept}"
        concept_out = output_root / f"{cls}_un{concept}"
        concept_out.mkdir(parents=True, exist_ok=True)

        # Collect candidate images
        images = sorted(
            [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        )
        if not images:
            tqdm.write(f"[info] No images found for class '{cls}' in {class_dir}")
            continue

        # Determine which images still need editing (resume logic)
        pending = [p for p in images if not (concept_out / f"{p.name}").exists()]
        if not pending:
            tqdm.write(
                f"[done] Nothing left to edit for Class='{cls}' Concept='{concept}'."
            )
            continue

        # ✅ Skip if pending < 10
        if len(pending) < 10:
            tqdm.write(
                f"[skip-small] Class='{cls}' Concept='{concept}' has only {len(pending)} pending (<10). Skipping batch."
            )
            continue

        # Create a clean staging directory for this batch with symlinks to 'pending'
        stage_dir = staging_root / f"{cls}__{concept}"
        if stage_dir.exists():
            # Clear previous staging content to avoid stale links
            for item in stage_dir.iterdir():
                try:
                    item.unlink()
                except Exception:
                    pass
        else:
            stage_dir.mkdir(parents=True, exist_ok=True)

        # Create symlinks so your script can take the whole directory at once
        for src in pending:
            link_path = stage_dir / src.name
            try:
                if link_path.exists():
                    link_path.unlink()
                os.symlink(src.resolve(), link_path)
            except (AttributeError, NotImplementedError, OSError):
                # Fallback to hardlink if symlink not available
                try:
                    if link_path.exists():
                        link_path.unlink()
                    os.link(src, link_path)
                except Exception as e:
                    tqdm.write(f"[warn] Could not link {src.name}: {e}")

        tqdm.write(
            f"[run] Class='{cls}' Concept='{concept}' | total={len(images)} pending={len(pending)} | "
            f"out -> {concept_out}"
        )

        # Single subprocess call per (Class, Concept) with the staging directory
        cmd = [
            sys.executable,
            str(script_path),
            "--prompt",
            prompt,
            "--model",
            "gpti1",
            "--edit_image_path",
            str(stage_dir),
            "--output_dir",
            str(concept_out),
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            tqdm.write(
                f"[error] batch failed for Class='{cls}' Concept='{concept}' (exit {e.returncode}). Continuing…"
            )
        except Exception as e:
            tqdm.write(
                f"[error] batch failed for Class='{cls}' Concept='{concept}': {e}. Continuing…"
            )
        finally:
            # Optional: clean up staging to keep things tidy
            try:
                for item in stage_dir.iterdir():
                    item.unlink()
                stage_dir.rmdir()
            except Exception:
                # If something holds the dir, we'll leave it; it’s safe to reuse/overwrite next run.
                pass


if __name__ == "__main__":
    run_concept_stripping_from_df_batched(
        script_path=str(Path(__file__).parent / "generative_pipelines.py"),
        df=pd.read_csv(Path(__file__).parent / "concept_prompts.csv"),
        classes_root="/home/tecnarca/PycharmProjects/Visual-TCAV/VisualTCAV/test_images",
        output_root="edited_images",
    )
