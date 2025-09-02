import argparse
import re
import numpy as np
import pandas as pd
from src.loggers import REPO_ROOT_PATH, LOGGER
from src.plotting.plotter import plot_similarity_vs_attribution_grouped



def load_prettytables_as_dataframe(file_path):
    # Read the content of the file
    text_content = file_path.read_text()
    # Use the previously loaded content to extract structured data into a DataFrame
    lines = [line.strip() for line in text_content.strip().split("\n") if "|" in line]

    # Find all blocks of tables separated by model/class headers
    tables = []
    current_model = current_class = current_examples = ""
    data_rows = []

    for line in lines:
        if "Model:" in line and "Class:" in line:
            if data_rows:
                tables.append(
                    (current_model, current_class, current_examples, data_rows)
                )
                data_rows = []
            parts = line.strip("|").split(";")
            current_model = parts[0].split(":")[1].strip()
            current_class = parts[1].split(":")[1].strip()
            current_examples = parts[2].split(":")[1].strip() if len(parts) > 2 else ""
        elif "+-" in line or "[" in line:
            parts = [part.strip() for part in line.strip("|").split("|")]
            if parts[0]:
                current_concept = parts[0]
            else:
                parts[0] = current_concept
            # Parse mean and std
            mean_std_match = re.match(r"([\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", parts[2])
            mean, std = (
                (float(mean_std_match.group(1)), float(mean_std_match.group(2)))
                if mean_std_match
                else (None, None)
            )
            # Parse CI
            ci_bounds = eval(parts[3])
            ci_low, ci_high = float(ci_bounds[0]), float(ci_bounds[1])
            data_rows.append(
                [
                    current_model,
                    current_class,
                    current_examples,
                    parts[0],
                    parts[1],
                    mean,
                    std,
                    ci_low,
                    ci_high,
                ]
            )

    # Append the last table
    if data_rows:
        tables.append((current_model, current_class, current_examples, data_rows))

    # Flatten all rows into one dataframe
    all_rows = [row for (_, _, _, table_rows) in tables for row in table_rows]
    df = pd.DataFrame(
        all_rows,
        columns=[
            "Model",
            "Class",
            "Examples",
            "Concept",
            "Layer",
            "Mean",
            "Std",
            "CI Low",
            "CI High",
        ],
    )
    return df


def load_dataframe_from_experiment_set(set_prefix: str) -> pd.DataFrame:
    root_dir = REPO_ROOT_PATH / "experiments"

    result = {
        folder.name: [
            f.name for f in folder.iterdir() if f.is_file() and f.suffix == ""
        ]
        for folder in root_dir.iterdir()
        if folder.is_dir() and folder.name.startswith(set_prefix)
    }

    loaded_dataframes = {
        (folder, file): load_prettytables_as_dataframe(root_dir / folder / file)
        for folder, files in result.items()
        for file in files
    }

    concats = pd.concat(
        [
            merged.assign(Folder=folder, file=file)
            for (folder, file), merged in loaded_dataframes.items()
        ],
        ignore_index=True,
    )
    concats = concats[concats["Concept"] != "random"]
    return concats.drop(columns=["CI Low", "CI High", "file"]).drop_duplicates()


def load_metrics_from_experiment_set(set_prefix: str) -> pd.DataFrame:
    root_dir = REPO_ROOT_PATH / "experiments"
    csv_files = sorted(root_dir.rglob("metrics.csv"))

    all_mergeds = []

    for file_path in csv_files:
        if str(file_path.parent.name).startswith(set_prefix):
            merged = pd.read_csv(file_path)
            merged["folder"] = file_path.parent.name  # keep source info
            all_mergeds.append(merged)

    concats = pd.concat(all_mergeds, ignore_index=True).fillna(0)
    concats.columns = [
        "example_file",
        "Model",
        "Layer",
        "Concept",
        "Similarity",
        "Folder",
    ]
    concats.drop(columns=["example_file"], inplace=True)
    concats = concats[concats["Concept"] != "random"]
    return concats.drop_duplicates()


def preprocess_and_clean(set_prefix: str, file_name="plotting_data.csv", also_save_data = True) -> pd.DataFrame:
    attributions = load_dataframe_from_experiment_set(set_prefix)
    similarities = load_metrics_from_experiment_set(set_prefix)
    merged = attributions.merge(
        similarities, on=["Model", "Concept", "Layer", "Folder"], how="left"
    )
    merged = merged.fillna({"Similarity": 1})  # true concepts have similarity 1 with themselves
    # keep only the last layer of each model
    latest_layers = merged.groupby("Model")["Layer"].max()
    merged = merged[merged["Layer"].isin(latest_layers)]
    merged["AblationOf"] = np.where(
        merged["Examples"] == merged["Class"],
        np.NAN,  # case when Examples == Class
        merged.apply(
            lambda row: row["Examples"].replace(f"{row['Class']}_un", "", 1),
            axis=1
        )
    )
    # --- Split parts ---
    merged["Concept_split"] = merged["Concept"].str.split("_").str[0]
    merged["LLM"] = merged["Concept"].str.split("_").str[1]
    merged.loc[
        merged.Concept.str.contains("taxi_sign") | merged.Concept.str.contains("leopard_print"), "LLM"
    ] = merged["Concept"].str.split("_").str[2]
    merged["Bootstrap"] = merged["Concept"].str.extract(r"_bootstrap_(\d+)")
    merged["Bootstrap"] = pd.to_numeric(merged["Bootstrap"], errors="coerce")
    merged["LLM"] = merged["LLM"].where(merged["Concept"].str.count("_") >= 2, np.nan)
    merged = merged.drop(columns=["Folder", "Layer", "Examples", "Concept"])
    merged = merged.rename(columns={"Concept_split": "Concept"})
    # --- Reorder columns ---
    categorical_cols = merged.select_dtypes(exclude=["number"]).columns.tolist()
    numerical_cols = merged.select_dtypes(include=["number"]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != "Bootstrap"]
    ordered_cols = categorical_cols + ["Bootstrap"] + numerical_cols
    merged = merged[ordered_cols]
    if also_save_data:
        merged.to_csv(LOGGER.experiment_path / file_name, index=False)
    return merged