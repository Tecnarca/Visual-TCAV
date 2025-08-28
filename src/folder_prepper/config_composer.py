#!/usr/bin/env python3
"""
Generate one YAML per class from a CSV of (Class, Concept).

- Reads CSV at: /home/tecnarca/PycharmProjects/Visual-TCAV/src/llm/concept_prompts.csv
  (override with --csv)
- For each Class, writes YAML to: /home/tecnarca/PycharmProjects/Visual-TCAV/configs/single_classes/<class>.yaml
  (override with --out)
- Uses example images and concept-imputation assets from:
    test_images:   /home/tecnarca/PycharmProjects/Visual-TCAV/VisualTCAV/test_images (override with --test)
    concept_images:/home/tecnarca/PycharmProjects/Visual-TCAV/VisualTCAV/concept_images (override with --concepts)

Rules:
- Class name is normalized to lowercase and spaces -> underscores for matching and YAML `name`.
- `example_image` must exist in `test_images` and be named like `<class>.<ext>`.
- For each (Class, Concept):
   * `true_label` is the (lowercased) Concept.
   * `generated` are the directory names in `concept_images` that equal the concept
     or start with `<concept>_` (e.g., `striped_flux`, `striped_sd35`, ...).
   * `concept_imputation.example` is an image file under `test_images` named either
       `<concept>_<class>.<ext>` OR `<class>_<concept>.<ext>` (first match wins).
   * `concept_imputation.folder` is a folder under `test_images` named either
       `<class>_<concept>` (preferred) or `<concept>_<class>`.
   * Both the example image and the folder must exist; otherwise that concept is skipped.

Outputs are merged with a static `models:` block as provided by the user.
"""

from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml  # PyYAML
except ImportError:
    yaml = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def normalize_name(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def find_example_image(test_root: Path, cls: str) -> Optional[str]:
    """Return basename of the representative image `<cls>.<ext>` in test_root, if present."""
    for p in sorted(test_root.iterdir()):
        if p.is_file() and p.stem == cls and p.suffix.lower() in IMAGE_EXTS:
            return p.name
    return None


def find_generated_variants(concepts_root: Path, concept: str) -> List[str]:
    """Return directory names in concepts_root that start with `concept_` (exclude exact match)."""
    out = []
    if not concepts_root.exists():
        return out
    for p in sorted(concepts_root.iterdir()):
        if p.is_dir():
            n = p.name.lower()
            if n.startswith(concept + "_"):
                out.append(p.name)  # keep original casing as on disk
    return out


def first_existing_image(test_root: Path, stems: List[str]) -> Optional[str]:
    """Return basename of the first image file whose stem matches any of `stems`."""
    candidates = []
    for stem in stems:
        for ext in IMAGE_EXTS:
            candidates.append(test_root / f"{stem}{ext}")
    for c in candidates:
        if c.exists() and c.is_file():
            return c.name
    return None


def pick_concept_folder(test_root: Path, cls: str, concept: str) -> Optional[str]:
    """Return folder name under test_root matching concept/class combos, including negated variants."""
    candidates = [
        f"{cls}_{concept}",
        f"{concept}_{cls}",
    ]
    for pre in ["un", "non", "no", "not", "anti"]:
        candidates.append(f"{cls}_{pre}{concept}")
        candidates.append(f"{pre}{concept}_{cls}")
    for name in candidates:
        if (test_root / name).is_dir():
            return name
    return None


def load_class_concepts(csv_path: Path) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = defaultdict(set)
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = {"Class", "Concept"}
        if not required.issubset(reader.fieldnames or {}):
            raise ValueError(f"CSV must contain columns {required}, found {reader.fieldnames}")
        for row in reader:
            cls_raw = row.get("Class", "").strip()
            concept_raw = row.get("Concept", "").strip()
            if not cls_raw or not concept_raw:
                continue
            cls = normalize_name(cls_raw)
            concept = normalize_name(concept_raw)
            mapping[cls].add(concept)
    return mapping


MODELS_BLOCK = [
    {
        "name": "ResNet50V2",
        "graph_path_filename": "ResNet50V2-architecture-and-weights-compiled.h5",
        "label_path_filename": "ResNet50V2-imagenet-classes.txt",
        "preprocessing_function": "tensorflow.keras.applications.resnet_v2.preprocess_input",
        "max_examples": 500,
        "layers": ["conv5_block1_out", "conv5_block2_out", "conv5_block3_out"],
    },
    {
        "name": "VGG16",
        "graph_path_filename": "VGG16-architecture-and-weights-compiled.h5",
        "label_path_filename": "VGG16-imagenet-classes.txt",
        "preprocessing_function": "tensorflow.keras.applications.vgg16.preprocess_input",
        "max_examples": 500,
        "layers": ["block5_conv1", "block5_conv2", "block5_conv3"],
    },
    {
        "name": "InceptionV3",
        "graph_path_filename": "InceptionV3-architecture-and-weights-compiled.h5",
        "label_path_filename": "InceptionV3-imagenet-classes.txt",
        "preprocessing_function": "tensorflow.keras.applications.inception_v3.preprocess_input",
        "max_examples": 500,
        "layers": ["mixed7", "mixed8", "mixed10"],
    },
    {
        "name": "ConvNeXt",
        "graph_path_filename": "ConvNeXt-architecture-and-weights-compiled",
        "label_path_filename": "ConvNeXt-imagenet-classes.txt",
        "preprocessing_function": "tensorflow.keras.applications.convnext.preprocess_input",
        "max_examples": 500,
        "layers": ['tf.__operators__.add_33', 'tf.__operators__.add_34', 'tf.__operators__.add_35'],
    },
]


def build_yaml_for_class(
    cls: str,
    concepts: Set[str],
    test_root: Path,
    concepts_root: Path,
) -> Optional[dict]:
    example_image = find_example_image(test_root, cls)
    if not example_image:
        print(f"⚠ Skipping class '{cls}': no example image '<cls>.<ext>' found in {test_root}")
        return None

    concept_groups = []
    for concept in sorted(concepts):
        generated = find_generated_variants(concepts_root, concept)
        stems = [f"{concept}_{cls}", f"{cls}_{concept}"]
        for pre in ["un", "non", "no", "not", "anti"]:
            stems.extend([f"{pre}{concept}_{cls}", f"{cls}_{pre}{concept}"])
        example_candidate = first_existing_image(test_root, stems)
        folder_name = pick_concept_folder(test_root, cls, concept)

        group = {
            "true_label": concept,
            "generated": generated,
        }
        if example_candidate and folder_name:
            group["concept_imputation"] = {
                "example": example_candidate,
                "folder": folder_name,
            }
        else:
            print(
                f"  ℹ No concept_imputation for concept '{concept}' in class '{cls}' "
                f"(looked for {stems})."
            )

        concept_groups.append(group)

    if not concept_groups:
        print(f"⚠ Skipping class '{cls}': no valid concepts found after checks.")
        return None

    return {
        "classes": [
            {
                "name": cls,
                "example_image": example_image,
                "concepts_groups": concept_groups,
            }
        ],
        "models": MODELS_BLOCK,
    }


def write_yaml(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        # Minimal fallback without PyYAML
        import json
        # Not true YAML, but provides a readable output; user can install PyYAML for proper YAML
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# Install PyYAML for canonical YAML output: pip install pyyaml\n")
            f.write(json.dumps(obj, indent=2))
        return
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)



def validate_yaml_paths(data: dict, test_root: Path, concepts_root: Path) -> List[str]:
    issues: List[str] = []
    for cls_entry in data.get("classes", []):
        ex = cls_entry.get("example_image")
        if ex:
            p = test_root / ex
            if not p.exists() or not p.is_file():
                issues.append(f"Missing example_image: {p}")
        for grp in cls_entry.get("concepts_groups", []):
            for gen in grp.get("generated", []):
                gp = concepts_root / gen
                if not gp.exists() or not gp.is_dir():
                    issues.append(f"Missing generated dir: {gp}")
            ci = grp.get("concept_imputation") or {}
            if ci:
                exf = test_root / ci.get("example", "")
                fld = test_root / ci.get("folder", "")
                if not exf.exists() or not exf.is_file():
                    issues.append(f"Missing concept_imputation.example: {exf}")
                if not fld.exists() or not fld.is_dir():
                    issues.append(f"Missing concept_imputation.folder: {fld}")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Generate single-class YAML configs from CSV")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/src/llm/concept_prompts.csv"),
        help="Path to concept_prompts.csv (with columns Class, Concept)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/configs/single_classes"),
        help="Output directory for generated YAML files",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("/VisualTCAV/test_images"),
        help="Root of test_images",
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        default=Path("/VisualTCAV/concept_images"),
        help="Root of concept_images",
    )

    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.test.exists():
        raise FileNotFoundError(f"test_images path not found: {args.test}")
    if not args.concepts.exists():
        print(f"⚠ concept_images path not found: {args.concepts}. 'generated' lists will be empty.")

    mapping = load_class_concepts(args.csv)
    if not mapping:
        print("No (Class, Concept) pairs found in CSV. Nothing to do.")
        return

    for cls, concepts in sorted(mapping.items()):
        data = build_yaml_for_class(cls, concepts, args.test, args.concepts)
        if data is None:
            continue
        out_file = args.out / f"{cls}.yaml"
        write_yaml(data, out_file)
        print(f"Wrote {out_file}")
        issues = validate_yaml_paths(data, args.test, args.concepts)
        if issues:
            print(f"Verification issues for {out_file}:")
            for m in issues:
                print(f"  - {m}")
        else:
            print(f"Verified all referenced paths for {out_file}.")


if __name__ == "__main__":
    main()
