import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# IMPORTANT: import the module AND the classes, so we can replace the global logger used elsewhere.
import src.loggers as loggers
from src.loggers import REPO_ROOT_PATH, ExperimentLogger  # noqa: E402
from src.visual_tcav.config_loader import load_config
from src.visual_tcav.framework.run_visual_tcav import (run_global_visual_tcav,
                                                       run_local_visual_tcav)

CHECKPOINTS_DIR = REPO_ROOT_PATH / ".tcav_checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def init_logger(base_name: str | None) -> ExperimentLogger:
    """
    Reinitialize the global logger used across the package, so other modules that
    import src.loggers.LOGGER will use the same run directory/base name.
    """

    return loggers.LOGGER


def checkpoint_path(base_name_or_fallback: str) -> Path:
    """
    Where we store progress to resume after a crash. We key by base_name so
    different runs don't collide.
    """
    safe_name = base_name_or_fallback.replace("/", "_").replace("\\", "_")
    return CHECKPOINTS_DIR / f"{safe_name}.json"


def load_checkpoint(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            # Corrupted checkpoint; start fresh but don't crash.
            return {"completed": []}
    return {"completed": []}


def save_checkpoint(path: Path, data: Dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def step_id(
    object_class_name: str,
    concept_group_label: str,
    model_name: str,
    action: str,
    target: str,
) -> str:
    return "|".join(
        [object_class_name, concept_group_label, model_name, action, target]
    )


def build_steps(config) -> List[Dict]:
    """
    Expand the config into an ordered list of atomic steps to run.
    Each step has:
      - id: stable identifier for checkpointing
      - run(): callable to execute the step
      - log: short description
    """
    steps: List[Dict] = []

    for object_class in config.classes:
        obj_name = object_class.name
        for concept_group in object_class.concepts_groups:
            cg_label = concept_group.true_label
            for model in config.models:
                model_name = model.name

                # 1) Local TCAV on the class example image
                if getattr(object_class, "example_image", None):
                    tgt = str(object_class.example_image)
                    steps.append(
                        {
                            "id": step_id(obj_name, cg_label, model_name, "local", tgt),
                            "log": f"Local • class example • {obj_name} • {cg_label} • {model_name}",
                            "run": lambda oc=object_class, cg=concept_group, m=model: run_local_visual_tcav(
                                oc.example_image, cg, m
                            ),
                        }
                    )

                # 2) Global TCAV on the object class
                steps.append(
                    {
                        "id": step_id(
                            obj_name, cg_label, model_name, "global", obj_name
                        ),
                        "log": f"Global • class • {obj_name} • {cg_label} • {model_name}",
                        "run": lambda oc=object_class, cg=concept_group, m=model: run_global_visual_tcav(
                            oc.name, oc, cg, m
                        ),
                    }
                )

                # 3) Optional: Local TCAV on concept imputation example
                imputation = getattr(concept_group, "concept_imputation", None)
                if imputation and getattr(imputation, "example", None):
                    tgt = str(imputation.example)
                    steps.append(
                        {
                            "id": step_id(
                                obj_name,
                                cg_label,
                                model_name,
                                "local_imputation_example",
                                tgt,
                            ),
                            "log": f"Local • imputation example • {obj_name} • {cg_label} • {model_name}",
                            "run": lambda cg=concept_group, m=model: run_local_visual_tcav(
                                cg.concept_imputation.example, cg, m
                            ),
                        }
                    )

                # 4) Optional: Global TCAV on concept imputation folder
                if imputation and getattr(imputation, "folder", None):
                    tgt = str(imputation.folder)
                    steps.append(
                        {
                            "id": step_id(
                                obj_name,
                                cg_label,
                                model_name,
                                "global_imputation_folder",
                                tgt,
                            ),
                            "log": f"Global • imputation folder • {obj_name} • {cg_label} • {model_name}",
                            "run": lambda oc=object_class, cg=concept_group, m=model: run_global_visual_tcav(
                                cg.concept_imputation.folder, oc, cg, m
                            ),
                        }
                    )
    return steps


def main(config_path: str, base_name: str | None):
    LOGGER = init_logger(base_name)
    config = load_config(config_path)

    # If the logger supports dumping config, keep that behavior:
    if hasattr(LOGGER, "log_config"):
        LOGGER.log_config(config.model_dump())

    if hasattr(LOGGER, "log_text"):
        run_name_info = getattr(LOGGER, "base_name", base_name) or "default"
        LOGGER.log_text(f"Running on config: {config_path}")
        LOGGER.log_text(f"Run base name: {run_name_info}")

    # Build flat step plan
    steps = build_steps(config)

    # Determine checkpoint path (prefer the active logger base_name if available)
    base_for_ckpt = getattr(LOGGER, "base_name", None) or base_name or "test_runs"
    ckpt_file = checkpoint_path(base_for_ckpt)
    ckpt = load_checkpoint(ckpt_file)
    completed: set[str] = set(ckpt.get("completed", []))

    # Count only steps we still need to run
    remaining = [s for s in steps if s["id"] not in completed]

    # Progress bar across ALL steps, pre-filled with completed count
    pbar = tqdm(
        total=len(steps), initial=len(completed), desc="Visual TCAV", unit="step"
    )

    for s in steps:
        if s["id"] in completed:
            # Already done in a previous run
            continue

        # Log step start
        if hasattr(LOGGER, "log_text"):
            LOGGER.log_text(f"▶ {s['log']}")

        try:
            s["run"]()  # run atomic step
        except SystemExit:
            # Let argparse or deliberate exits propagate (but persist progress first)
            ckpt["last_failed"] = s["id"]
            save_checkpoint(ckpt_file, ckpt)
            raise
        except Exception as e:
            # Record failure and exit so a rerun can resume from here.
            ckpt["last_failed"] = s["id"]
            save_checkpoint(ckpt_file, ckpt)
            if hasattr(LOGGER, "log_text"):
                LOGGER.log_text(f"✖ FAILED: {s['log']} with {type(e).__name__}: {e}")
            # Provide traceback to stderr for quick debugging.
            print(f"\nERROR in step {s['id']}: {e}", file=sys.stderr)
            raise  # crash-safe: on next invocation we resume from this step
        else:
            # Mark as completed and persist
            completed.add(s["id"])
            ckpt["completed"] = sorted(completed)
            ckpt.pop("last_failed", None)
            save_checkpoint(ckpt_file, ckpt)
            if hasattr(LOGGER, "log_text"):
                LOGGER.log_text(f"✓ DONE: {s['log']}")
        finally:
            pbar.update(1)

    pbar.close()
    if hasattr(LOGGER, "log_text"):
        LOGGER.log_text("All steps completed successfully ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Visual TCAV analysis using a config file."
    )
    parser.add_argument(
        "config_path",
        type=str,
        nargs="?",
        default=str(REPO_ROOT_PATH / "configs/config_minimal.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default=None,
        help="Override ExperimentLogger base run name (affects output dir and checkpoint).",
    )
    args = parser.parse_args()

    main(args.config_path, args.base_name)
