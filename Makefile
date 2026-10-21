.PHONY: all install run debug clean clean-venv clean-data lint lint-strict data help

PYTHON ?= uv run python
ARGS ?= --help

all: install

help:
	@echo "Available targets:"
	@echo "  install      Install project dependencies using uv"
	@echo "  run          Execute the main entrypoint (uv run python -m src)"
	@echo "  debug        Execute the main entrypoint in debug mode using pdb"
	@echo "  clean        Remove temporary cache files and directories"
	@echo "  clean-data   Remove the data/ directory completely"
	@echo "  lint         Run flake8 and mypy according to subject requirements"
	@echo "  lint-strict  Run flake8 and mypy in strict mode"
	@echo "  data         Create data/ structure and extract vLLM/dataset zip files"

install:
	uv sync

data:
	@echo "Creating directory structure..."
	@mkdir -p data/raw data/datasets
	@echo "Extracting datasets..."
	@for f in *dataset*.zip; do [ -f "$$f" ] && (unzip -q -o "$$f" -d data/datasets/ 2>/dev/null || python3 -m zipfile -e "$$f" data/datasets/ || true); done
	@if [ -d data/datasets/datasets_public/public ]; then mv data/datasets/datasets_public/public/* data/datasets/ && rm -rf data/datasets/datasets_public; fi
	@echo "[OK] Datasets extracted."
	@echo "Extracting vLLM (~3,200 files, please wait ~20-30s in WSL)..."
	@for f in *vllm*.zip; do [ -f "$$f" ] && (unzip -q -o "$$f" -d data/raw/ 2>/dev/null || python3 -m zipfile -e "$$f" data/raw/ || true); done
	@echo "[OK] vLLM extracted."
	@echo "Data extraction complete."

clean-data:
	@echo "Removing data/ directory..."
	@rm -rf data
	@echo "Data directory removed."

run:
	$(PYTHON) -m src $(ARGS)

debug:
	$(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache .ruff_cache

clean-venv:
	rm -rf .venv

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

