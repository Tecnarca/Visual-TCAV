SRC_DIR = src
CHECK_DIRS= $(SRC_DIR)
.PHONY: format
CONFIG ?= configs/config_minimal.yaml

format: ## Format repository code
	uv run black $(CHECK_DIRS)
	uv run isort $(CHECK_DIRS)

.PHONY: install
install: ## Install the dependencies from the lock file
	uv sync -v

.PHONY: run
run: ## Run the visual tcav analysis
	uv run python -m src.visual_tcav.main $(CONFIG)

.PHONY: run-all-classes

run-all-classes:
	@for file in configs/classes/*.yaml; do \
		echo "Running: python -m src.visual_tcav.main $$file"; \
		make run $$file; \
	done

.PHONY: help
help: ## Show the available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
