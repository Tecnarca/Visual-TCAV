import argparse

from src.loggers import LOGGER
from src.plotting.plot_data_loader import preprocess_and_clean
from src.plotting.plotter import plot_similarity_vs_attribution_grouped

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_experiment_set",
        type=str,
        default="test_runs",
        help="Prefix of the experiment set to load (default: test_runs)"
    )
    args = parser.parse_args()

    graph_data = preprocess_and_clean(args.from_experiment_set)

    plot_similarity_vs_attribution_grouped(graph_data, LOGGER.experiment_path / "images")