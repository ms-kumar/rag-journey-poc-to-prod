.PHONY: clean format check lint run ingest test help sync install test-cov test-all type-check security quality pre-commit dev eval-datasets eval eval-ci dashboard eval-full

# Python and project settings
PYTHON := uv run python
SRC_DIR := src
TESTS_DIR := tests

help:
	@echo "Available targets:"
	@echo "  make sync         - Sync dependencies with uv"
	@echo "  make install      - Install all dependencies including dev extras"
	@echo "  make run          - Run the FastAPI server"
	@echo "  make dev          - Run server with auto-reload"
	@echo "  make ingest       - Ingest documents via API"
	@echo "  make test         - Run tests (excluding slow tests)"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make test-all     - Run all tests including slow ones"
	@echo "  make format       - Format code with ruff"
	@echo "  make check        - Check code formatting with ruff"
	@echo "  make lint         - Lint code with ruff"
	@echo "  make lint-fix     - Lint and auto-fix issues"
	@echo "  make type-check   - Run mypy type checking"
	@echo "  make security     - Run bandit security checks"
	@echo "  make quality      - Run all quality checks"
	@echo "  make pre-commit   - Setup pre-commit hooks"
	@echo "  make clean        - Remove cache and build artifacts"
	@echo ""
	@echo "Evaluation targets:"
	@echo "  make eval-datasets - Create evaluation datasets"
	@echo "  make eval          - Run evaluation with default dataset"
	@echo "  make eval-ci       - Run CI evaluation gate"
	@echo "  make dashboard     - Generate evaluation dashboard"
	@echo "  make eval-full     - Run full evaluation workflow"

# Sync dependencies with uv
sync:
	@echo "Syncing dependencies with uv..."
	uv sync --extra dev --no-dev

# Install all dependencies including dev extras
install:
	@echo "Installing all dependencies..."
	uv sync --all-extras

# Run the server (with dependency check)
run: sync
	uv run python -m src.main

# Run server with auto-reload for development
dev: sync
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

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

# Run type checking with mypy
type-check:
	uv run mypy --namespace-packages -p src

# Run security checks with bandit
security:
	uv run --extra dev bandit -r $(SRC_DIR) -c pyproject.toml

# Run all quality checks
quality: check lint type-check security
	@echo "✅ All quality checks passed!"

# Setup pre-commit hooks
pre-commit:
	uv run pre-commit install
	@echo "✅ Pre-commit hooks installed!"

# Run all checks (format check + lint)
all-checks: check lint

# Clean up cache files and build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/ .eggs/ 2>/dev/null || true

# Run tests (excluding slow tests)
test: sync
	uv run python -m pytest $(TESTS_DIR) -v -k "not slow"

# Run tests with coverage
test-cov: sync
	uv run python -m pytest $(TESTS_DIR) -v -k "not slow" --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "📊 Coverage report generated in htmlcov/index.html"

# Run all tests including slow ones
test-all: sync
	uv run python -m pytest $(TESTS_DIR) -v

# Create evaluation datasets
eval-datasets: sync
	@echo "Creating evaluation datasets..."
	uv run python scripts/create_eval_datasets.py

# Create sample evaluation results for dashboard demo
eval-samples: sync
	@echo "Creating sample evaluation results..."
	uv run python scripts/create_sample_results.py

# Run evaluation with default dataset
eval: sync
	@echo "Running evaluation..."
	uv run python scripts/ci_eval_gate.py --dataset data/eval/rag_default_eval.json

# Run CI evaluation gate (quick test)
eval-ci: sync
	@echo "Running CI evaluation gate..."
	uv run python scripts/ci_eval_gate.py --dataset data/eval/rag_test_small.json

# Generate evaluation dashboard
dashboard: sync
	@echo "Generating evaluation dashboard..."
	uv run python scripts/generate_dashboard.py
	@echo "✅ Dashboard generated: results/dashboard.html"

# Run full evaluation workflow
eval-full: eval-datasets eval dashboard
	@echo "✅ Full evaluation workflow completed!"
