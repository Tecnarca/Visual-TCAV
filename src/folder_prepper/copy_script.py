#!/usr/bin/env python3
"""
Copy each subfolder from `edited_images` to `VisualTCAV/test_images` with a normalized name,
then copy one image from each subfolder into `test_images` itself named after the (normalized) folder.

Normalization rules:
- lower-case
- spaces -> underscores

Defaults are derived from your paths but can be overridden via CLI args.

Example:
    python copy_and_extract_test_images.py \
        --src "/home/tecnarca/PycharmProjects/Visual-TCAV/edited_images" \
        --dst "/home/tecnarca/PycharmProjects/Visual-TCAV/VisualTCAV/test_images"
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def first_image_in(folder: Path) -> Path | None:
    # Look only at files directly in the folder (not recursive), sort for determinism
    for p in sorted(folder.iterdir()):
        if is_image(p):
            return p
    # If none at top level, try recursively (optional):
    for p in sorted(folder.rglob("*")):
        if is_image(p):
            return p
    return None


def unique_path(base: Path) -> Path:
    """If base exists, append _1, _2, ... before the suffix until unique."""
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    i = 1
    while True:
        candidate = base.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def copy_subfolders_and_images(src_root: Path, dst_root: Path) -> None:
    if not src_root.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    # Iterate only immediate subdirectories of src_root
    subdirs: Iterable[Path] = [p for p in src_root.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"No subfolders found under {src_root}")
        return

    for src_subdir in sorted(subdirs):
        norm_name = normalize_name(src_subdir.name)
        dst_subdir = dst_root / norm_name

        # Copy/merge folder (Python 3.8+: dirs_exist_ok)
        shutil.copytree(src_subdir, dst_subdir, dirs_exist_ok=True)
        print(f"Copied folder: {src_subdir} -> {dst_subdir}")
        one_img = first_image_in(dst_subdir)
        if one_img is None:
            print(f"  ⚠ No images found inside '{src_subdir.name}'. Skipping representative image.")
            continue
        img_ext = one_img.suffix.lower()
        rep_name = f"{norm_name}{img_ext}"
        rep_dst = unique_path(dst_root / rep_name)
        shutil.copy2(one_img, rep_dst)
        print(f"  Extracted image: {one_img.name} -> {rep_dst.name}")
        """
        one_img = first_image_in(dst_subdir)
        if one_img is None:
            print(f"  ⚠ No images found inside '{src_subdir.name}'. Skipping representative image.")
            continue
        img_ext = one_img.suffix.lower()
        rep_name = f"{norm_name}{img_ext}"
        rep_dst = unique_path(dst_root / rep_name)
        shutil.copy2(one_img, rep_dst)
        print(f"  Extracted image: {one_img.name} -> {rep_dst.name}")
        """


def parse_args() -> argparse.Namespace:
    default_src = "/edited_images"
    default_dst = "/VisualTCAV/test_images"

    #default_src = "/home/tecnarca/PycharmProjects/Visual-TCAV/generated_images"
    #default_dst = "/home/tecnarca/PycharmProjects/Visual-TCAV/VisualTCAV/concept_images"

    ap = argparse.ArgumentParser(description="Copy normalized folders and extract one image each.")
    ap.add_argument("--src", type=Path, default=Path(default_src),
                    help="Source root containing subfolders (default: %(default)s)")
    ap.add_argument("--dst", type=Path, default=Path(default_dst),
                    help="Destination root for copied folders and images (default: %(default)s)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    copy_subfolders_and_images(args.src, args.dst)
