import os
import random
import shutil
from pathlib import Path

from src.loggers import REPO_ROOT_PATH


def bootstrap_concept(concept_name, n=5, ratio=1):
    """
    Create `n` bootstrap folders from the original `concept_name`, sampling files
    with replacement and writing *symlinks* to the originals instead of copies.

    Args:
        concept_name (str | Path): Name of the concept folder under VisualTCAV/concept_images.
        n (int): Number of bootstrap folders to create.
        ratio (float): Multiplier controlling size of each sample relative to the original dataset.
    """
    base = REPO_ROOT_PATH / "VisualTCAV" / "concept_images"
    concept_path = base / concept_name
    files = [p for p in concept_path.iterdir() if p.is_file()]
    if not files:
        raise ValueError(f"No files found in folder: {concept_path}")

    sample_size = max(1, int(len(files) * ratio))
    bootstrapped_folders = []

    for i in range(n):
        new_folder = concept_path.parent / f"{concept_path.name}_bootstrap_{i}"
        bootstrapped_folders.append(new_folder.name)

        if new_folder.exists():
            shutil.rmtree(new_folder)
        new_folder.mkdir(parents=True, exist_ok=True)

        # Sample with replacement
        sample = [random.choice(files) for _ in range(sample_size)]

        for src in sample:
            # Start with the original name; add suffixes if a name collision occurs
            target = new_folder / src.name
            counter = 1
            while target.exists():
                target = new_folder / f"{src.stem}_{counter}{src.suffix}"
                counter += 1

            # Use a relative path for the symlink to keep things portable
            rel_src = os.path.relpath(src, start=new_folder)
            try:
                target.symlink_to(
                    rel_src
                )  # files only; no need for target_is_directory=True
            except OSError as e:
                # Common on Windows without Developer Mode/admin. Be explicit.
                raise OSError(
                    f"Failed to create symlink '{target}' -> '{rel_src}'. "
                    "On Windows, enable Developer Mode or run with admin privileges."
                ) from e

    return bootstrapped_folders


def delete_bootstrap_folders():
    """
    Delete every bootstrap folder under REPO_ROOT_PATH / "VisualTCAV" / "concept_images".

    A bootstrap folder is identified by having "_bootstrap_" in its name.
    """
    base = REPO_ROOT_PATH / "VisualTCAV" / "concept_images"
    if not base.exists():
        raise FileNotFoundError(f"Base path does not exist: {base}")

    deleted = []
    for folder in base.iterdir():
        if folder.is_dir() and "_bootstrap_" in folder.name:
            shutil.rmtree(folder)
            deleted.append(folder.name)

    return deleted
