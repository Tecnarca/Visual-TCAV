SRC_DIR = src
CHECK_DIRS= $(SRC_DIR)
.PHONY: format
CONFIG ?= configs/config_minimal.yaml

format: ## Format repository code
	uv run black $(CHECK_DIRS)
	uv run isort $(CHECK_DIRS)

.PHONY: install
install: ## Install the dependencies from the lock file
	uv sync -v --prerelease=allow

.PHONY: run
run: ## Run the visual tcav analysis
	uv run python -m src.visual_tcav.main $(CONFIG)

.PHONY: run-all-classes

FILES := $(wildcard configs/single_classes/*.yaml)

run-all-classes:
	@for file in $(FILES); do \
	  base=$$(basename "$$file" .yaml); \
	  echo "Running: uv run python -m src.visual_tcav.main \"$$file\" --base-name \"$$base\""; \
	  uv run python -m src.visual_tcav.main "$$file" --base-name "$$base"; \
	done

.PHONY: help
help: ## Show the available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
