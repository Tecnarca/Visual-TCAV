import pandas as pd
import matplotlib.pyplot as plt
from src.loggers import REPO_ROOT_PATH, LOGGER

# Set this to True if you want to display mean and std instead of grouped bars
display_with_variance = True

if __name__ == "__main__":
    root_dir = REPO_ROOT_PATH / "experiments"
    print(f"Root directory: {root_dir.resolve()}")

    # Find all metrics.csv files
    csv_files = list(root_dir.rglob("metrics.csv"))
    csv_files = sorted(csv_files)

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        source_name = file_path.parent.name  # Folder name as label

        for step, df_step in df.groupby("step"):
            if display_with_variance:
                # Compute mean and std across key_1 for each key_2
                grouped = df_step.groupby("key_2")["value"]
                mean_series = grouped.mean()
                std_series = 2 * grouped.std()

                ax = mean_series.plot(
                    kind="bar",
                    yerr=std_series,
                    capsize=4,
                    figsize=(10, 6),
                    color="skyblue",
                    ecolor="black",
                    error_kw=dict(lw=1, ls='--')
                )

                ax.set_title(f"{source_name} - Model: {step} (Mean ± 2*Std)")
                ax.set_ylabel("Cosine similarity to ground truth")
                ax.set_xlabel("Concept name")
                plt.ylim(0, 1)
                plt.xticks(rotation=45)
                plt.tight_layout()

                LOGGER.log_figure(plt, "local_cav_analysis")
                plt.show()

            else:
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
