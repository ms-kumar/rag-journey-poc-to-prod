.PHONY: clean format check lint run ingest test help sync install test-cov test-all type-check security quality pre-commit dev eval-datasets eval eval-ci dashboard eval-full test-canary test-adversarial test-guardrails test-violation-threshold guardrails-report guardrails-audit-review test-agent test-cache test-cost test-embeddings test-evaluation test-experimentation test-retrieval test-ingestion test-performance test-observability docker-build docker-push deploy-staging deploy-canary deploy-prod rollback canary-health rehearse-rollback

# Python and project settings
PYTHON := uv run python
SRC_DIR := src
TESTS_DIR := tests

# Docker/Deployment settings
IMAGE_NAME ?= rag-api
IMAGE_TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "latest")
REGISTRY ?= ghcr.io/$(shell git config --get remote.origin.url | sed 's/.*[:/]\([^/]*\/[^.]*\).*/\1/' 2>/dev/null || echo "owner/repo")
NAMESPACE_STAGING ?= staging
NAMESPACE_PROD ?= production

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
	@echo "Module-specific test targets:"
	@echo "  make test-agent       - Run agent module tests"
	@echo "  make test-cache       - Run cache module tests"
	@echo "  make test-cost        - Run cost module tests"
	@echo "  make test-embeddings  - Run embeddings module tests"
	@echo "  make test-evaluation  - Run evaluation module tests"
	@echo "  make test-guardrails  - Run guardrails module tests"
	@echo "  make test-observability - Run observability module tests"
	@echo "  make test-experimentation - Run experimentation module tests"
	@echo "  make test-retrieval   - Run retrieval module tests"
	@echo "  make test-ingestion   - Run ingestion module tests"
	@echo "  make test-performance - Run performance module tests"
	@echo ""
	@echo "Evaluation targets:"
	@echo "  make eval-datasets - Create evaluation datasets"
	@echo "  make eval          - Run evaluation with default dataset"
	@echo "  make eval-ci       - Run CI evaluation gate"
	@echo "  make dashboard     - Generate evaluation dashboard"
	@echo "  make eval-full     - Run full evaluation workflow"
	@echo ""
	@echo "Guardrails & Security targets:"
	@echo "  make test-canary              - Run quick canary tests (CI)"
	@echo "  make test-adversarial         - Run adversarial/red-team tests"
	@echo "  make test-violation-threshold - Verify violation rate ≤ 0.1%"
	@echo "  make guardrails-report        - Generate compliance report"
	@echo "  make guardrails-audit-review  - Review audit logs"
	@echo ""
	@echo "Docker & Deployment targets:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-push    - Build and push image to registry"
	@echo "  make deploy-staging - Deploy to staging environment"
	@echo "  make deploy-canary  - Deploy canary (5% traffic)"
	@echo "  make deploy-prod    - Deploy to production (full rollout)"
	@echo "  make rollback ENV=x - Rollback deployment (staging|production)"
	@echo "  make canary-health  - Check canary health metrics"
	@echo "  make rehearse-rollback - Run rollback rehearsal"
	@echo "  make deploy-status  - Show deployment status"
	@echo "  make deploy-history ENV=x - View deployment history"

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
	@echo "Running tests in batches to avoid memory issues..."
	@uv run python -m pytest $(TESTS_DIR)/unit/services/agent -p no:cacheprovider --tb=line -q || true
	@uv run python -m pytest $(TESTS_DIR)/unit/services/cache -p no:cacheprovider --tb=line -q || true
	@uv run python -m pytest $(TESTS_DIR)/unit/services/cost -p no:cacheprovider --tb=line -q || true
	@uv run python -m pytest $(TESTS_DIR)/unit/services/embeddings -p no:cacheprovider --tb=line -q || true
	@uv run python -m pytest $(TESTS_DIR)/unit/services/observability -p no:cacheprovider --tb=line -q || true
	@uv run python -m pytest $(TESTS_DIR)/unit/services/experimentation -p no:cacheprovider --tb=line -q || true
	@uv run python -m pytest $(TESTS_DIR)/unit/services/retrieval -p no:cacheprovider --tb=line -q || true
	@uv run python -m pytest $(TESTS_DIR)/unit/services/performance -p no:cacheprovider --tb=line -q || true
	@uv run python -m pytest $(TESTS_DIR)/unit/test_*.py -p no:cacheprovider --tb=line -q || true
	@uv run python -m pytest $(TESTS_DIR)/integration -p no:cacheprovider --tb=line -q \
		--ignore=$(TESTS_DIR)/integration/test_sandbox_integration.py || true
	@echo "✅ Test execution complete (some tests may have failed)"

# Module-specific test targets
test-agent: sync
	@echo "Running agent module tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/agent/ -v

test-cache: sync
	@echo "Running cache module tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/cache/ -v

test-embeddings: sync
	@echo "Running embeddings module tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/embeddings/ -v

test-evaluation: sync
	@echo "Running evaluation module tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/evaluation/ -v

test-retrieval: sync
	@echo "Running retrieval module tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/retrieval/ -v

test-ingestion: sync
	@echo "Running ingestion module tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/ingestion/ -v

test-performance: sync
	@echo "Running performance module tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/performance/ -v

test-observability: sync
	@echo "Running observability module tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/observability/ -v

test-experimentation: sync
	@echo "Running experimentation module tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/experimentation/ -v

test-cost: sync
	@echo "Running cost module tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/cost/ -v

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
# Guardrails and adversarial testing targets
# Run canary tests (quick smoke tests for CI)
test-canary: sync
	@echo "Running canary tests for guardrails..."
	uv run python -m pytest -m canary $(TESTS_DIR)/unit/services/guardrails/test_adversarial_guardrails.py -v
	@echo "✅ Canary tests passed!"

# Run adversarial tests (red-team prompts, jailbreak tests)
test-adversarial: sync
	@echo "Running adversarial tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/guardrails/test_adversarial_guardrails.py -v
	@echo "✅ Adversarial tests passed!"

# Run all guardrails tests
test-guardrails: sync
	@echo "Running all guardrails tests..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/guardrails/ -v
	@echo "✅ All guardrails tests passed!"

# Verify violation thresholds (critical test)
test-violation-threshold: sync
	@echo "Verifying violation rate ≤ 0.1%..."
	uv run python -m pytest $(TESTS_DIR)/unit/services/guardrails/test_adversarial_guardrails.py::TestAdversarialGuardrails::test_overall_adversarial_violation_rate -v
	@echo "✅ Violation threshold check passed!"

# Generate guardrails compliance report
guardrails-report: sync
	@echo "Generating guardrails compliance report..."
	@echo "Recent violations:"
	@if [ -f logs/audit.log ]; then \
		tail -n 100 logs/audit.log | grep -i "violation" || echo "No recent violations found"; \
	else \
		echo "Audit log not found"; \
	fi

# Review recent audit logs
guardrails-audit-review: sync
	@echo "Recent guardrails audit events:"
	@if [ -f logs/audit.log ]; then \
		tail -n 50 logs/audit.log; \
	else \
		echo "Audit log not found"; \
	fi

# ============================================
# Docker & Deployment Targets
# ============================================

# Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -t $(IMAGE_NAME):latest .
	@echo "✅ Built $(IMAGE_NAME):$(IMAGE_TAG)"

# Push Docker image to registry
docker-push: docker-build
	@echo "🚀 Pushing Docker image to $(REGISTRY)..."
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker tag $(IMAGE_NAME):latest $(REGISTRY)/$(IMAGE_NAME):latest
	docker push $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest
	@echo "✅ Pushed to $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)"

# Deploy to staging environment
deploy-staging: docker-push
	@echo "🎯 Deploying to staging..."
	kubectl -n $(NAMESPACE_STAGING) set image deployment/rag-api \
		rag-api=$(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	kubectl -n $(NAMESPACE_STAGING) rollout status deployment/rag-api --timeout=5m
	@echo "✅ Deployed to staging!"

# Deploy canary (5% traffic)
deploy-canary:
	@echo "🐤 Deploying canary..."
	kubectl -n $(NAMESPACE_PROD) set image deployment/rag-api-canary \
		rag-api=$(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	kubectl -n $(NAMESPACE_PROD) scale deployment/rag-api-canary --replicas=1
	@echo "✅ Canary deployed! Monitor metrics before promoting."

# Deploy to production (full rollout)
deploy-prod:
	@echo "🚀 Deploying to production..."
	@echo "⚠️  This will deploy to all production pods!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	kubectl -n $(NAMESPACE_PROD) set image deployment/rag-api \
		rag-api=$(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	kubectl -n $(NAMESPACE_PROD) rollout status deployment/rag-api --timeout=10m
	@echo "✅ Production deployment complete!"

# Rollback to previous version
rollback:
	@echo "⏪ Rolling back..."
ifndef ENV
	$(error ENV is required. Usage: make rollback ENV=staging|production)
endif
ifeq ($(ENV),staging)
	kubectl -n $(NAMESPACE_STAGING) rollout undo deployment/rag-api
	kubectl -n $(NAMESPACE_STAGING) rollout status deployment/rag-api
else ifeq ($(ENV),production)
	kubectl -n $(NAMESPACE_PROD) rollout undo deployment/rag-api
	kubectl -n $(NAMESPACE_PROD) rollout status deployment/rag-api
else
	$(error Invalid ENV. Use staging or production)
endif
	@echo "✅ Rollback complete!"

# Check canary health
canary-health:
	@echo "🔍 Checking canary health..."
	uv run python scripts/check_canary_health.py \
		--deployment rag-api-canary \
		--error-threshold 0.05 \
		--latency-threshold 500 \
		--min-requests 100

# Run rollback rehearsal
rehearse-rollback:
	@echo "🎭 Running rollback rehearsal..."
	uv run python scripts/rehearse_rollback.py \
		--scenario error_spike \
		--environment staging \
		--output results/rollback_rehearsal_$(shell date +%Y%m%d_%H%M%S).json

# Show deployment status
deploy-status:
	@echo "📊 Deployment Status:"
	@echo "\n=== Staging ==="
	@kubectl -n $(NAMESPACE_STAGING) get deployment rag-api -o wide 2>/dev/null || echo "Not found"
	@echo "\n=== Production ==="
	@kubectl -n $(NAMESPACE_PROD) get deployment rag-api -o wide 2>/dev/null || echo "Not found"
	@echo "\n=== Canary ==="
	@kubectl -n $(NAMESPACE_PROD) get deployment rag-api-canary -o wide 2>/dev/null || echo "Not found"

# View deployment history
deploy-history:
	@echo "📜 Deployment History:"
ifndef ENV
	$(error ENV is required. Usage: make deploy-history ENV=staging|production)
endif
ifeq ($(ENV),staging)
	kubectl -n $(NAMESPACE_STAGING) rollout history deployment/rag-api
else ifeq ($(ENV),production)
	kubectl -n $(NAMESPACE_PROD) rollout history deployment/rag-api
endif