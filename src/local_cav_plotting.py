import pandas as pd
import matplotlib.pyplot as plt
from src.loggers import REPO_ROOT_PATH, LOGGER

if __name__ == "__main__":
    root_dir = REPO_ROOT_PATH / "experiments"
    print(f"Root directory: {root_dir.resolve()}")

    # Find all metrics.csv files
    csv_files = list(root_dir.rglob("metrics.csv"))
    csv_files = sorted(csv_files)

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        source_name = file_path.parent.name  # Folder name as label

        # Group by step and plot each one
        for step, df_step in df.groupby("step"):
            # Pivot the table to get key_2 on X and key_1 as columns
            pivot_df = df_step.pivot(index="key_2", columns="key_1", values="value")

            ax = pivot_df.plot(kind="bar", figsize=(10, 6))
            ax.set_title(f"{source_name} - Model: {step}")
            ax.set_ylabel("Cosine similarity to ground truth")
            ax.set_xlabel("Concept name")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.legend(title="Layer")
            plt.tight_layout()

            LOGGER.log_figure(plt, "local_cav_analysis")

            plt.show()
