import random
import shutil
from pathlib import Path

from src.loggers import REPO_ROOT_PATH


def boostrap_concept(concept_name, n=5, ratio=1):
    """
    Create `n` bootstrap folders from the original `folder`, sampling files with replacement.

    Args:
        concept_name (str or Path): Path to the original folder containing files to sample.
        n (int): Number of bootstrap folders to create.
        ratio (float): Multiplier to control size of each sample relative to the original dataset.
    """
    concept_name = REPO_ROOT_PATH / "VisualTCAV" / "concept_images" / concept_name
    files = list(concept_name.iterdir())
    if not files:
        raise ValueError(f"No files found in folder: {concept_name}")

    sample_size = max(1, int(len(files) * ratio))

    boostraped_folders = []

    for i in range(n):
        new_folder = concept_name.parent / f"{concept_name.name}_bootstrap_{i}"
        boostraped_folders.append(new_folder.name)
        if new_folder.exists():
            shutil.rmtree(new_folder)
        new_folder.mkdir(exist_ok=True)
        sample = [random.choice(files) for _ in range(sample_size)]

        for f in sample:
            target_path = new_folder / f.name
            counter = 1
            while target_path.exists():
                target_path = new_folder / f"{f.stem}_{counter}{f.suffix}"
                counter += 1
            shutil.copy2(f, target_path)
    return boostraped_folders
