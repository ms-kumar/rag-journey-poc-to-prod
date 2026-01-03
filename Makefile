.PHONY: clean format check lint run ingest test help sync

# Python and project settings
PYTHON := uv run python
SRC_DIR := src
TESTS_DIR := tests

help:
	@echo "Available targets:"
	@echo "  make sync      - Sync dependencies with uv"
	@echo "  make run       - Run the FastAPI server"
	@echo "  make ingest    - Ingest documents via API"
	@echo "  make format    - Format code with ruff"
	@echo "  make check     - Check code formatting with ruff"
	@echo "  make lint      - Lint code with ruff"
	@echo "  make clean     - Remove cache and build artifacts"
	@echo "  make test      - Run tests"

# Sync dependencies with uv
sync:
	@echo "Syncing dependencies with uv..."
	uv sync

# Run the server (with dependency check)
run: sync
	uv run python -m src.main

# Ingest documents
ingest:
	curl -X POST http://localhost:8000/api/v1/rag/ingest

# Format code with ruff
format:
	uv run ruff format $(SRC_DIR) $(TESTS_DIR)

# Check formatting without making changes
check:
	uv run ruff format --check $(SRC_DIR) $(TESTS_DIR)

# Lint code with ruff
lint:
	uv run ruff check $(SRC_DIR) $(TESTS_DIR)

# Lint and auto-fix
lint-fix:
	uv run ruff check --fix $(SRC_DIR) $(TESTS_DIR)

# Run all checks (format check + lint)
all-checks: check lint

# Clean up cache files and build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/ .eggs/ 2>/dev/null || true

# Run tests (with dependency check)
test: sync
	uv run pytest $(TESTS_DIR) -v

# Install dependencies
install:
	pip install -e ".[dev]"
